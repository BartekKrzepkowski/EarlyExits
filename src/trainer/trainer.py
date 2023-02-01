import os
import datetime
from collections import defaultdict

import torch
import torch.nn.functional as F
from tqdm import tqdm, trange

import wandb
from logger import WandbLogger
from utils import adjust_dict, update_dict
# from trainers.tensorboard_pytorch import TensorboardPyTorch


class EarlyExitTrainer(object):
    def __init__(self, ee_method, loaders, criterion, optim, accelerator=None, lr_scheduler=None, device='cpu'):
        self.ee_method = ee_method
        self.criterion = criterion
        self.optim = optim
        self.accelerator = accelerator
        self.lr_scheduler = lr_scheduler
        self.loaders = loaders
        self.t_logger = None  # tensorflow logger
        self.device = device

    def run_exp(self, epoch_start, epoch_end, exp_name, config_run_epoch, temp=1.0, random_seed=42):
        """
        Main method of trainer.
        Args:
            epoch_start (int): A number representing the beginning of run
            epoch_end (int): A number representing the end of run
            exp_name (str): Base name of experiment
            config_run_epoch (): ##
            temp (float): CrossEntropy Temperature
            random_seed (int): Seed generator
        """
        save_path = self.at_exp_start(exp_name, random_seed)
        for epoch in trange(epoch_start, epoch_end, desc='run_exp'):
            wandb.log({'epoch': epoch}, step=epoch)
            self.ee_method.train()
            self.run_epoch(epoch, save_path, config_run_epoch, phase='train')
            self.ee_method.eval()
            with torch.no_grad():
                self.run_epoch(epoch, save_path, config_run_epoch, phase='test')
        wandb.finish()

    def at_exp_start(self, exp_name, random_seed):
        """
        Initialization of experiment.
        Creates fullname, dirs and loggers.
        Args:
            exp_name (str): Base name of experiment
            random_seed (int): seed generator
        Returns:
            save_path (str): Path to save the model
        """
        self.manual_seed(random_seed)
        print('is fp16?', self.accelerator.use_fp16)
        date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_path = os.path.join(os.getcwd(), f'exps/{exp_name}/{date}')
        save_path = f'{base_path}/checkpoints'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        WandbLogger(wandb, exp_name, 0)
        # wandb.tensorboard.patch(root_logdir=f'{base_path}/tensorboard', pytorch=True, save=True)
        # wandb.watch(self.ee_method, log_freq=1000, idx=0, log_graph=True, log='all', criterion=self.criterion)
        # self.t_logger = TensorboardPyTorch(f'{base_path}/tensorboard', self.device)
        return save_path

    def run_epoch(self, epoch, save_path, config_run_epoch, phase):
        """
        Init df -> [Init Run -> [Run Epoch]_{IL} -> Update df]_{IL}]
        {IL - In Loop}
        Args:
            epoch (int): Current epoch
            save_path (str): Path to save the model
            config_run_epoch (): ##
            phase (train|test): Phase of training
        """
        epoch_loss = 0.0
        epoch_correct = 0.0 # w przypadku train correct z ostatniej warstwy, w p.p. correct z warstw wyjścia
        epoch_denom = 0.0
        running_loss = 0.0
        running_correct = 0.0
        running_denom = 0.0
        if phase == 'train':
            running_bcs_loss = {}
            running_bcs_correct = {}
            epoch_bcs_loss = {}
            epoch_bcs_correct = {}

        loader_size = len(self.loaders[phase])
        global_step = epoch * loader_size
        progress_bar = tqdm(self.loaders[phase], desc=f'run_epoch: {phase}',
                            mininterval=30, leave=False, total=loader_size)
        for i, data in enumerate(progress_bar):
            global_step += 1
            wandb.log({f'step/{phase}': i}, step=global_step)
            x_true, y_true = data
            # with torch.autocast(device_type=self.device, dtype=torch.float16):
            if 'train' in phase:
                loss, correct, bcs_loss, bcs_correct = self.ee_method.run_train(x_true, y_true)
                loss /= config_run_epoch.grad_accum_steps
                self.accelerator.backward(loss) # jedyne użycie acceleratora w trainerze, wraz z clip_grad_norm
                if (i + 1) % config_run_epoch.grad_accum_steps == 0 or (i + 1) == loader_size:
                    # self.accelerator.clip_grad_norm_(filter(lambda p: p.requires_grad, self.ee_method.parameters()), 3.0)
                    self.optim.step()
                    if self.lr_scheduler is not None: # na pewno w tym miejscu?
                        self.lr_scheduler.step()
                    self.optim.zero_grad()
                loss *= config_run_epoch.grad_accum_steps
                running_bcs_loss = update_dict(running_bcs_loss, bcs_loss)
                running_bcs_correct = update_dict(running_bcs_correct, bcs_correct)
            else:
                loss, correct = self.ee_method.run_val(x_true, y_true)


            denom = y_true.size(0)
            running_loss += loss.item() * denom
            running_correct += correct
            running_denom += denom
            # loggers
            if (i + 1) % config_run_epoch.grad_accum_steps * config_run_epoch.running_step_mult == 0 or (i + 1) == loader_size:
                tmp_loss = running_loss / running_denom
                tmp_acc = running_correct / running_denom
                losses = {f'running_{phase}/ee_loss': round(tmp_loss, 4), f'running_{phase}/ee_acc': round(tmp_acc, 4)}
                progress_bar.set_postfix(losses)
                wandb.log(losses, step=global_step)

                # self.t_logger.log_scalar(f'running_loss/{phase}', losses[f'running_loss/{phase}'], global_step)
                # self.t_logger.log_scalar(f'running_acc/{phase}', losses[f'running_acc/{phase}'], global_step)

                if phase == 'train':
                    adjusted_running_bcs_loss = adjust_dict(running_bcs_loss, running_denom, 'loss', 'running')
                    adjusted_running_bcs_correct = adjust_dict(running_bcs_correct, running_denom, 'acc', 'running')
                    wandb.log(adjusted_running_bcs_loss, step=global_step)
                    wandb.log(adjusted_running_bcs_correct, step=global_step)
                    epoch_bcs_loss = update_dict(epoch_bcs_loss, running_bcs_loss)
                    epoch_bcs_correct = update_dict(epoch_bcs_correct, running_bcs_correct)
                    running_bcs_loss = {}
                    running_bcs_correct = {}

                epoch_loss += running_loss
                epoch_correct += running_correct
                epoch_denom += running_denom
                running_loss = 0.0
                running_correct = 0.0
                running_denom = 0.0

                # if (i + 1) % config_run_epoch.save_interval == 0 or (i + 1) == loader_size:
                #     self.save_model(save_path, i)

                if (i + 1) == loader_size:
                    tmp_loss = epoch_loss / epoch_denom
                    tmp_acc = epoch_correct / epoch_denom
                    global_losses = {f'epoch_{phase}/ee_loss': round(tmp_loss, 4),
                                     f'epoch_{phase}/ee_acc': round(tmp_acc, 4)}
                    # t_logger.log_scalar(f'epoch_loss/{phase}', global_losses[f'epoch_loss/{phase}'], epoch)
                    # t_logger.log_scalar(f'epoch_acc/{phase}', global_losses[f'epoch_acc/{phase}'], epoch)
                    wandb.log(global_losses, step=epoch)
                    if phase == 'train':
                        adjusted_epoch_bcs_loss = adjust_dict(epoch_bcs_loss, epoch_denom, 'loss', 'epoch')
                        adjusted_epoch_bcs_acc = adjust_dict(epoch_bcs_correct, epoch_denom, 'acc', 'epoch')
                        wandb.log(adjusted_epoch_bcs_loss, step=epoch)
                        wandb.log(adjusted_epoch_bcs_acc, step=epoch)

                if self.lr_scheduler is not None and phase == 'train':
                    # t_logger.log_scalar('lr_scheduler', self.lr_scheduler.get_last_lr()[0], global_step)
                    wandb.log({'lr_scheduler': self.lr_scheduler.get_last_lr()[0]}, step=global_step)


    def save_model(self, path, step):
        """
        Save the model.
        Args:
            save_path (str): Path to save the model
            step (int): Step of the run
        """
        torch.save(self.ee_method.state_dict(), f"{path}/model_{datetime.datetime.utcnow()}_step_{step}.pth")
        # self.student.save_pretrained(f"{path}/model_{datetime.datetime.utcnow()}_step_{step}.pth")

    def manual_seed(self, random_seed):
        """
        Set the environment for reproducibility purposes.
        Args:
            random_seed (int): seed generator
        """
        import random
        import numpy as np
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if 'cuda' in self.device.type:
            torch.cuda.empty_cache()
            # torch.backends.cudnn.deterministic = True
            torch.cuda.manual_seed_all(random_seed)
            # torch.backends.cudnn.benchmark = False
