from collections import defaultdict

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
        Creates fullname, dirs and loggers.
        Args:
            exp_name (str): Base name of experiment
            random_seed (int): seed generator
        """
        self.manual_seed(config)
        base_path, save_path = create_paths(config.base_path, config.exp_name)
        self.base_path, self.save_path = base_path, save_path
        config.logger_config['log_dir'] = f'{base_path}/{config.logger_name}'
        if config.logger_name == 'tensorboard':
            config.logger_config['layout'] = ee_tensorboard_layout(self.ee_wrapper.n_layers)
        self.logger = LOGGERS_NAME_MAP[config.logger_name](config)

    def run_epoch(self, phase, config):
        self.epoch_evaluators = defaultdict(float)
        self.epoch_denom = 0.0
        running_evaluators = defaultdict(float)
        running_denom = 0.0

        if phase == 'train':
            self.epoch_bcs_loss = defaultdict(float) # zamien na defaultdict
            self.epoch_bcs_correct = defaultdict(float) #
            running_bcs_loss = defaultdict(float)
            running_bcs_correct = defaultdict(float)
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
                loss, evaluators, bcs_loss, bcs_correct = self.ee_wrapper.run_train(x_true, y_true)
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
                running_bcs_loss = adjust_evaluators(running_bcs_loss, bcs_loss, denom,
                                                     'running_loss_ce_per_layer/layer', phase)
                running_bcs_correct = adjust_evaluators(running_bcs_correct, bcs_correct, denom,
                                                        'running_acc_per_layer/layer', phase)
            else:
                evaluators, sample_exited_at = self.ee_wrapper.run_val(x_true, y_true)
                running_sample_exited_at = update_tensor(running_sample_exited_at, sample_exited_at)

            running_evaluators = adjust_evaluators(running_evaluators, evaluators, denom, 'running', phase)
            running_denom += denom

            if (i + 1) % (config.grad_accum_steps * config.step_multi) == 0 or (i + 1) == loader_size:
                whether_to_log_running = (i + 1) % (config.grad_accum_steps * config.step_multi) == 0
                runnings = {
                    'evaluators': running_evaluators,
                    'denom': running_denom,
                    'bcs_loss': running_bcs_loss if phase == 'train' else None,
                    'bcs_correct': running_bcs_correct if phase == 'train' else None,
                    'sample_exited_at': sample_exited_at if phase != 'train' else None
                }
                self.log(runnings, phase, i, progress_bar, whether_to_log_running)
                running_evaluators = defaultdict(float)
                running_denom = 0.0
                if phase == 'train':
                    running_bcs_loss = defaultdict(float)
                    running_bcs_correct = defaultdict(float)
                else:
                    running_sample_exited_at = torch.Tensor([])

    def log(self, runnings, phase, i, progress_bar, whether_to_log_running):
        self.epoch_evaluators = adjust_evaluators(self.epoch_evaluators, runnings['evaluators'], 1, 'epoch', phase)
        self.epoch_denom += runnings['denom']

        if whether_to_log_running:
            running_evaluators_log = adjust_evaluators_pre_log(runnings['evaluators'], runnings['denom'], round_at=4)
            self.logger.log_scalars(running_evaluators_log, self.global_step)
            progress_bar.set_postfix(running_evaluators_log)

            if phase == 'train':
                running_bcs_loss_log = adjust_evaluators_pre_log(runnings['bcs_loss'], runnings['denom'], round_at=4)
                running_bcs_acc_log = adjust_evaluators_pre_log(runnings['bcs_correct'], runnings['denom'], round_at=4)
                self.logger.log_scalars(running_bcs_loss_log, global_step=self.global_step)
                self.logger.log_scalars(running_bcs_acc_log, global_step=self.global_step)
                self.epoch_bcs_loss = adjust_evaluators(self.epoch_bcs_loss, runnings['bcs_loss'], 1, 'epoch', phase)
                self.epoch_bcs_correct = adjust_evaluators(self.epoch_bcs_correct, runnings['bcs_correct'], 1,
                                                           'epoch', phase)
            else:
                self.logger.log_histogram(f'running_sample_exited_at ({phase})', runnings['sample_exited_at'],
                                          global_step=self.global_step)
                self.epoch_sample_exited_at = update_tensor(self.epoch_sample_exited_at, runnings['sample_exited_at'])

        loader_size = progress_bar.total
        if (i + 1) == loader_size:
            epoch_evaluators_log = adjust_evaluators_pre_log(self.epoch_evaluators, self.epoch_denom, round_at=4)
            self.logger.log_scalars(epoch_evaluators_log, self.epoch)
            progress_bar.set_postfix(epoch_evaluators_log)
            #
            if phase == 'train':
                epoch_bcs_loss_log = adjust_evaluators_pre_log(self.epoch_bcs_loss, self.epoch_denom, round_at=4)
                epoch_bcs_acc_log = adjust_evaluators_pre_log(self.epoch_bcs_correct, self.epoch_denom, round_at=4)
                self.logger.log_scalars(epoch_bcs_loss_log, global_step=self.epoch)
                self.logger.log_scalars(epoch_bcs_acc_log, global_step=self.epoch)
            else:
                self.logger.log_histogram(f'epoch_sample_exited_at ({phase})', self.epoch_sample_exited_at,
                                          global_step=self.epoch)

        if self.lr_scheduler is not None and phase == 'train':
            self.logger.log_scalars({f'lr_scheduler': self.lr_scheduler.get_last_lr()[0]}, self.global_step)

    def update_metrics(self, metrics, phase):
        self.epoch_evaluators = adjust_evaluators(self.epoch_evaluators, metrics['evaluators'], 1, 'epoch', phase)
        self.epoch_denom += metrics['denom']
        if phase == 'train':
            self.epoch_bcs_loss = adjust_evaluators(self.epoch_bcs_loss, metrics['bcs_loss'], 1, 'epoch', phase)
            self.epoch_bcs_correct = adjust_evaluators(self.epoch_bcs_correct, metrics['bcs_correct'], 1,
                                                       'epoch', phase)
        else:
            self.epoch_sample_exited_at = update_tensor(self.epoch_sample_exited_at, metrics['sample_exited_at'])

    def manual_seed(self, config):
        """
        Set the environment for reproducibility purposes.
        Args:
            random_seed (int): seed generator
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
