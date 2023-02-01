from collections import defaultdict
from typing import List, Tuple, Dict, Union

import torch
from tqdm import tqdm, trange

from src.common.common import LOGGERS_NAME_MAP, ee_tensorboard_layout
from src.common.utils import adjust_evaluators, adjust_evaluators_pre_log, create_paths, update_tensor


class EarlyExitTrainer:
    def __init__(self, ee_wrapper, loaders, optim, lr_scheduler):
        self.ee_wrapper = ee_wrapper
        self.loaders = loaders
        self.optim = optim
        self.lr_scheduler = lr_scheduler

        self.logger = None
        self.base_path = None
        self.save_path = None
        self.epoch = None
        self.epoch_evaluators = None
        self.epoch_ics_loss = None
        self.epoch_ics_correct = None
        self.epoch_sample_exited_at = None
        self.epoch_denom = None
        self.global_step = None

    def run_exp(self, config):
        """
        Main method of trainer.
        Args:
            config (dict): Consists of:
                epoch_start (int): A number representing the beginning of run
                epoch_end (int): A number representing the end of run
                grad_accum_steps (int):
                step_multi (int):
                base_path (str): Base path
                exp_name (str): Base name of experiment
                logger_name (str): Logger type
                random_seed (int): Seed generator
        """
        self.at_exp_start(config)
        for epoch in tqdm(range(config.epoch_start_at, config.epoch_end_at), desc='run_exp'):
            self.epoch = epoch
            self.ee_wrapper.train()
            self.run_epoch(phase='train', config=config)
            self.ee_wrapper.eval()
            with torch.no_grad():
                self.run_epoch(phase='test', config=config)
            self.logger.close()

    def at_exp_start(self, config):
        """
        Initialization of experiment.
        Creates fullname, dirs and logger.
        """
        self.manual_seed(config)
        base_path, save_path = create_paths(config.base_path, config.exp_name)
        self.base_path, self.save_path = base_path, save_path
        config.logger_config['log_dir'] = f'{base_path}/{config.logger_name}'
        if config.logger_name == 'tensorboard':
            config.logger_config['layout'] = ee_tensorboard_layout(self.ee_wrapper.n_layers)
        self.logger = LOGGERS_NAME_MAP[config.logger_name](config)

    def run_epoch(self, phase, config):
        '''
        Run single epoch
        Args:
            phase (str): phase of the trening
            config (dict):
        '''
        self.epoch_evaluators = defaultdict(float)
        self.epoch_denom = 0.0
        running_evaluators = defaultdict(float)
        running_denom = 0.0

        if phase == 'train':
            self.epoch_ics_loss = defaultdict(float)
            self.epoch_ics_correct = defaultdict(float)
            running_ics_loss = defaultdict(float)
            running_ics_correct = defaultdict(float)
        else:
            self.epoch_sample_exited_at = torch.Tensor([])
            running_sample_exited_at = torch.Tensor([])

        loader_size = len(self.loaders[phase])
        progress_bar = tqdm(self.loaders[phase], desc=f'run_epoch: {phase}', mininterval=30,
                            leave=True, total=loader_size, position=0, ncols=200) # ruszałem
        self.global_step = self.epoch * loader_size
        for i, data in enumerate(progress_bar):
            self.global_step += 1
            x_true, y_true = data
            x_true, y_true = x_true.to(config.device), y_true.to(config.device)
            denom = y_true.size(0)
            if 'train' in phase:
                loss, evaluators, ics_loss, ics_correct = self.ee_wrapper.run_train(x_true, y_true)
                loss /= config.grad_accum_steps
                loss.backward()
                if (i + 1) % config.grad_accum_steps == 0 or (i + 1) == loader_size:
                    if config.whether_clip:
                        torch.nn.utils.clip_grad_norm(filter(lambda p: p.requires_grad, self.ee_wrapper.parameters()),
                                                      config.clip_value)
                    self.optim.step()
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
                    self.optim.zero_grad()
                loss *= config.grad_accum_steps
                running_ics_loss = adjust_evaluators(running_ics_loss, ics_loss, denom,
                                                     'running_loss_ce_per_layer/layer', phase)
                running_ics_correct = adjust_evaluators(running_ics_correct, ics_correct, denom,
                                                        'running_acc_per_layer/layer', phase)
            else:
                evaluators, sample_exited_at = self.ee_wrapper.run_val(x_true, y_true)
                running_sample_exited_at = update_tensor(running_sample_exited_at, sample_exited_at)

            running_evaluators = adjust_evaluators(running_evaluators, evaluators, denom, 'running', phase)
            running_denom += denom

            whether_log = (i + 1) % (config.grad_accum_steps * config.step_multi) == 0
            whether_epoch_end = (i + 1) == loader_size

            if whether_log or whether_epoch_end:
                running_assets = {
                    'evaluators': running_evaluators,
                    'denom': running_denom,
                    'ics_loss': running_ics_loss if phase == 'train' else None,
                    'ics_correct': running_ics_correct if phase == 'train' else None,
                    'sample_exited_at': running_sample_exited_at if phase != 'train' else None
                }
                self.update_epoch_assets(running_assets, phase)

            if whether_log:
                self.log(running_assets, phase, 'running', progress_bar)
                running_evaluators = defaultdict(float)
                running_denom = 0.0
                if phase == 'train':
                    running_ics_loss = defaultdict(float)
                    running_ics_correct = defaultdict(float)
                else:
                    running_sample_exited_at = torch.Tensor([])

            if whether_epoch_end:
                epoch_assets = {
                    'evaluators': self.epoch_evaluators,
                    'denom': self.epoch_denom,
                    'ics_loss': self.epoch_ics_loss if phase == 'train' else None,
                    'ics_correct': self.epoch_ics_correct if phase == 'train' else None,
                    'sample_exited_at': self.epoch_sample_exited_at if phase != 'train' else None
                }
                self.log(epoch_assets, phase, 'epoch', progress_bar)

    def log(self, assets: Dict, phase: str, scope: str, progress_bar: tqdm):
        '''
        Send chosen assets to logger and progress bar
        Args:
            assets (Dict):
            phase:
            scope:
            progress_bar:
        '''
        evaluators_log = adjust_evaluators_pre_log(assets['evaluators'], assets['denom'], round_at=4)
        self.logger.log_scalars(evaluators_log, self.global_step)
        progress_bar.set_postfix(evaluators_log)

        if phase == 'train':
            ics_loss_log = adjust_evaluators_pre_log(assets['ics_loss'], assets['denom'], round_at=4)
            ics_acc_log = adjust_evaluators_pre_log(assets['ics_correct'], assets['denom'], round_at=4)
            self.logger.log_scalars(ics_loss_log, global_step=self.global_step)
            self.logger.log_scalars(ics_acc_log, global_step=self.global_step)
        else:
            self.logger.log_histogram(f'{scope}_sample_exited_at ({phase})', assets['sample_exited_at'],
                                      global_step=self.global_step)

        if self.lr_scheduler is not None and phase == 'train' and scope == 'running':
            self.logger.log_scalars({f'lr_scheduler': self.lr_scheduler.get_last_lr()[0]}, self.global_step)

    def update_epoch_assets(self, assets: Dict, phase: str):
        '''
        Update epoch assets
        Args:
            assets (Dict): set of running metrics
            phase (str): phase of the trening
        '''
        self.epoch_evaluators = adjust_evaluators(self.epoch_evaluators, assets['evaluators'], 1, 'epoch', phase)
        self.epoch_denom += assets['denom']
        if phase == 'train':
            self.epoch_ics_loss = adjust_evaluators(self.epoch_ics_loss, assets['ics_loss'], 1, 'epoch', phase)
            self.epoch_ics_correct = adjust_evaluators(self.epoch_ics_correct, assets['ics_correct'], 1,
                                                       'epoch', phase)
        else:
            self.epoch_sample_exited_at = update_tensor(self.epoch_sample_exited_at, assets['sample_exited_at'])

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
