from collections import defaultdict
from typing import List, Tuple, Dict, Union

import torch
from tqdm import tqdm, trange

from src.common.common import LOGGERS_NAME_MAP, ee_tensorboard_layout
from src.common.utils import adjust_evaluators, adjust_evaluators_pre_log, create_paths, update_tensor, \
    save_model, clip_grad_norm


class EarlyExitTrainer:
    def __init__(self, ee_wrapper, loaders, optim, lr_scheduler, accelerator):
        self.ee_wrapper = ee_wrapper
        self.loaders = loaders
        self.optim = optim
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

        self.logger = None
        self.base_path = None
        self.save_path = None
        self.epoch = None
        self.global_step = None

    def run_exp(self, config):
        """
        Main method of trainer.
        Set seed, run train-val in the loop.
        Args:
            config (dict): Consists of:
                epoch_start (int): A number representing the beginning of run
                epoch_end (int): A number representing the end of run
                grad_accum_steps (int): number of gradient accumulation steps
                save_multi (int): Multipler of grad_accum_steps, how many steps to save weight
                log_multi (int): Multipler of grad_accum_steps, how many steps to log
                whether_clip (bool): Whether to clip the norm of gradients?
                clip_value (float): Clip norm value
                base_path (str): Base path
                exp_name (str): Base name of experiment
                logger_name (str): Logger type
                logger_config (dict): Logger configuration
                random_seed (int): Seed generator
                device (torch.device): Whether cpu or gpu device
        """
        self.manual_seed(config)
        self.at_exp_start(config)
        for epoch in tqdm(range(config.epoch_start_at, config.epoch_end_at), desc='run_exp'):
            self.epoch = epoch
            self.ee_wrapper.train()
            self.run_epoch(phase='train', config=config)
            self.ee_wrapper.eval()
            with torch.no_grad():
                # self.run_epoch(phase='val', config=config)
                self.run_epoch(phase='test', config=config)
        self.logger.close()
        save_model(self.ee_wrapper, self.save_path(self.global_step))

    def at_exp_start(self, config):
        """
        Initialization of experiment.
        Creates fullname, dirs and logger.
        """
        self.base_path, self.save_path = create_paths(config.base_path, config.exp_name)
        config.logger_config['log_dir'] = f'{self.base_path}/{config.logger_name}'
        if config.logger_name == 'tensorboard':
            config.logger_config['layout'] = ee_tensorboard_layout(self.ee_wrapper.n_layers)
        self.logger = LOGGERS_NAME_MAP[config.logger_name](config)

    def run_epoch(self, phase, config):
        """
        Run single epoch
        Args:
            phase (str): phase of the traning
            config (dict): as in run_exp
        """
        running_assets = {
            'evaluators': defaultdict(float),
            'denom': 0.0,
            'ics_loss': defaultdict(float),
            'ics_correct': defaultdict(float),
            'sample_exited_at': torch.Tensor([])
        }
        epoch_assets = {
            'evaluators': defaultdict(float),
            'denom': 0.0,
            'ics_loss': defaultdict(float),
            'ics_correct': defaultdict(float),
            'sample_exited_at': torch.Tensor([])
        }
        loader_size = len(self.loaders[phase])
        progress_bar = tqdm(self.loaders[phase], desc=f'run_epoch: {phase}', mininterval=30,
                            leave=True, total=loader_size, position=0, ncols=150) # ruszałem
        self.global_step = self.epoch * loader_size
        for i, data in enumerate(progress_bar):
            self.global_step += 1
            x_true, y_true = data
            x_true, y_true = x_true.to(config.device), y_true.to(config.device)
            if phase == 'train':
                loss, evaluators, ics_loss, ics_correct = self.ee_wrapper.run_train(x_true, y_true)
                loss /= config.grad_accum_steps
                self.accelerator.backward(loss) if self.accelerator is not None else loss.backward()
                if (i + 1) % config.grad_accum_steps == 0 or (i + 1) == loader_size:
                    if config.whether_clip:
                        clip_grad_wrapper = self.accelerator.clip_grad_norm_ \
                            if self.accelerator is not None else torch.nn.utils.clip_grad_norm
                        clip_grad_norm(clip_grad_wrapper, self.ee_wrapper, config.clip_value)
                    self.optim.step()
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
                    self.optim.zero_grad()
                loss *= config.grad_accum_steps
            else:
                evaluators, sample_exited_at = self.ee_wrapper.run_val(x_true, y_true)

            step_assets = {
                'evaluators': evaluators,
                'denom': y_true.size(0),
                'ics_loss': ics_loss if phase == 'train' else None,
                'ics_correct': ics_correct if phase == 'train' else None,
                'sample_exited_at': sample_exited_at if phase != 'train' else None
            }
            running_assets = self.update_assets(running_assets, step_assets, step_assets['denom'], 'running', phase)

            whether_save_model = (i + 1) % (config.grad_accum_steps * config.save_multi) == 0
            whether_log = (i + 1) % (config.grad_accum_steps * config.log_multi) == 0
            whether_epoch_end = (i + 1) == loader_size

            if whether_save_model and 'train' in phase:
                save_model(self.ee_wrapper, self.save_path(self.global_step))

            if whether_log or whether_epoch_end:
                epoch_assets = self.update_assets(epoch_assets, running_assets, 1.0, 'epoch', phase)

            if whether_log:
                self.log(running_assets, 'running', phase, progress_bar, step=self.global_step)
                running_assets['evaluators'] = defaultdict(float)
                running_assets['denom'] = 0.0
                running_assets['ics_loss'] = defaultdict(float)
                running_assets['ics_correct'] = defaultdict(float)
                running_assets['sample_exited_at'] = torch.Tensor([])

            if whether_epoch_end:
                self.log(epoch_assets, 'epoch', phase, progress_bar, step=self.epoch)

    def log(self, assets: Dict, scope: str, phase: str, progress_bar: tqdm, step: int):
        '''
        Send chosen assets to logger and progress bar
        Args:
            assets (Dict): Assets to log
            scope (str): Either running or epoch
            phase (str): phase of the traning
            progress_bar: Progress bar
            step (int): Step to log
        '''
        evaluators_log = adjust_evaluators_pre_log(assets['evaluators'], assets['denom'], round_at=4)
        self.logger.log_scalars(evaluators_log, global_step=step)
        progress_bar.set_postfix(evaluators_log)

        if phase == 'train':
            ics_loss_log = adjust_evaluators_pre_log(assets['ics_loss'], assets['denom'], round_at=4)
            ics_acc_log = adjust_evaluators_pre_log(assets['ics_correct'], assets['denom'], round_at=4)
            self.logger.log_scalars(ics_loss_log, global_step=step)
            self.logger.log_scalars(ics_acc_log, global_step=step)
        else:
            self.logger.log_histogram(f'{scope}_sample_exited_at ({phase})', assets['sample_exited_at'],
                                      global_step=step)

        if self.lr_scheduler is not None and phase == 'train' and scope == 'running':
            self.logger.log_scalars({f'lr_scheduler': self.lr_scheduler.get_last_lr()[0]}, step)

    def update_assets(self, assets_target: Dict, assets_source: Dict, multiplier, scope, phase: str):
        '''
        Update epoch assets
        Args:
            assets_target (Dict): Assets to which assets should be transferred
            assets_source (Dict): Assets from which assets should be transferred
            multiplier (int): Number to get rid of the average
            scope (str): Either running or epoch
            phase (str): Phase of the traning
        '''
        assets_target['evaluators'] = adjust_evaluators(assets_target['evaluators'], assets_source['evaluators'],
                                                        multiplier, scope, phase)
        assets_target['denom'] += assets_source['denom']
        if phase == 'train':
            scope_loss = 'running_loss_ce_per_layer/layer' if scope == 'running' else scope
            scope_correct = 'running_acc_per_layer/layer' if scope == 'running' else scope
            assets_target['ics_loss'] = adjust_evaluators(assets_target['ics_loss'], assets_source['ics_loss'],
                                                          multiplier, scope_loss, phase)

            assets_target['ics_correct'] = adjust_evaluators(assets_target['ics_correct'], assets_source['ics_correct'],
                                                             multiplier, scope_correct, phase)
        else:
            assets_target['sample_exited_at'] = update_tensor(assets_target['sample_exited_at'],
                                                              assets_source['sample_exited_at'])
        return assets_target

    def manual_seed(self, config: defaultdict):
        """
        Set the environment for reproducibility purposes.
        Args:
            config (defaultdict): set of parameters
                usage of:
                    random seed (int):
                    device (torch.device):
        """
        import random
        import numpy as np
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)
        torch.manual_seed(config.random_seed)
        if 'cuda' in config.device.type:
            torch.cuda.empty_cache()
            torch.cuda.manual_seed_all(config.random_seed)
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False
