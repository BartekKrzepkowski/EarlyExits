import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

#model
from resnets_eff import ResNet34

model = ResNet34(num_classes=10)#.to(device)

PATH = 'model_2023-01-10 15:25:15.316290_step_0.pth'
model.load_state_dict(torch.load(PATH))

for p in model.parameters():
    p.requires_grad = False

from sdn import SDN
LD = [64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 256, 256, 512, 512, 512]
ee_method = SDN(backbone_model=model, layers_dim=LD, confidence_threshold=0.1, num_classes=10, device=device)

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

cifar_mean = (0.4914, 0.4822, 0.4465)
cifar_std = (0.2023, 0.1994, 0.2010)

transform_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
])
transform_train = transforms.Compose([
    transforms.TrivialAugmentWide(),
    transforms.ToTensor(),
    transforms.Normalize(cifar_mean, cifar_std),
    transforms.RandomErasing(p=0.1),
])

train_dataset = CIFAR10(root='.',
                        train=True,
                        download=True,
                        transform=transform_eval)

test_dataset = CIFAR10(root='.',
                       train=False,
                       download=True,
                       transform=transform_eval)

BATCH_SIZE = 64

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=8)

loaders = {
    'train': train_loader,
    'test': test_loader
}

criterion = torch.nn.CrossEntropyLoss().to(device)
optim = torch.optim.AdamW(ee_method.parameters(), **{'lr': 0.05, 'weight_decay': 0.001})
lr_scheduler = None

import os
import datetime
from tensorboard_pytorch import TensorboardPyTorch

exp_name = 'cifar10_resnet34_sdn'
date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
base_path = os.path.join(os.getcwd(), f'exps/{exp_name}/{date}')
save_path = f'{base_path}/checkpoints'
if not os.path.exists(save_path):
    os.makedirs(save_path)
t_logger = TensorboardPyTorch(f'{base_path}/tensorboard', device)

from tqdm import tqdm
from utils import adjust_dict, update_dict, update_tensor
GRAD_ACCUM_STEPS = 1
STEP_MULTI = 1

def run_epoch(epoch, phase):
    epoch_loss = 0.0
    epoch_correct = 0.0
    epoch_denom = 0.0
    running_loss = 0.0
    running_correct = 0.0
    running_denom = 0.0
    if phase == 'train':
        running_bcs_loss = {}
        running_bcs_correct = {}
        epoch_bcs_loss = {}
        epoch_bcs_correct = {}
    else:
        running_sample_exited_at = torch.Tensor([])
        epoch_sample_exited_at = torch.Tensor([])
    loader_size = len(loaders[phase])
    global_step = epoch * loader_size
    progress_bar = tqdm(loaders[phase], desc=f'run_epoch: {phase}',
                        mininterval=30, leave=False, total=loader_size)
    for i, data in enumerate(progress_bar):
        global_step += 1
        x_true, y_true = data
        x_true, y_true = x_true.to(device), y_true.to(device)
        if 'train' in phase:
            loss, correct, bcs_loss, bcs_correct = ee_method.run_train(x_true, y_true)
            loss /= GRAD_ACCUM_STEPS
            loss.backward()
            if (i + 1) % GRAD_ACCUM_STEPS == 0 or (i + 1) == loader_size:
                optim.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()
                optim.zero_grad()
            loss *= GRAD_ACCUM_STEPS
            running_bcs_loss = update_dict(running_bcs_loss, bcs_loss)
            running_bcs_correct = update_dict(running_bcs_correct, bcs_correct)
        else:
            loss, correct, sample_exited_at = ee_method.run_val(x_true, y_true)
            running_sample_exited_at = update_tensor(running_sample_exited_at, sample_exited_at)

        denom = y_true.size(0)
        running_correct += correct
        running_loss += loss.item() * denom
        running_denom += denom

        if (i + 1) % (GRAD_ACCUM_STEPS * STEP_MULTI) == 0 or (i + 1) == loader_size:
            tmp_loss = running_loss / running_denom
            tmp_acc = running_correct / running_denom
            losses = {f'running_loss/{phase}': round(tmp_loss, 4), f'running_acc/{phase}': round(tmp_acc, 4)}
            progress_bar.set_postfix(losses)

            t_logger.log_scalar(f'running_loss/{phase}', losses[f'running_loss/{phase}'], global_step)
            t_logger.log_scalar(f'running_acc/{phase}', losses[f'running_acc/{phase}'], global_step)

            if phase == 'train':
                adjusted_running_bcs_loss = adjust_dict(running_bcs_loss, running_denom, 'loss', 'running')
                adjusted_running_bcs_acc = adjust_dict(running_bcs_correct, running_denom, 'acc', 'running')
                t_logger.log_scalars(f'running_loss_per_layer ({phase})', adjusted_running_bcs_loss, global_step=global_step)
                t_logger.log_scalars(f'running_acc_per_layer ({phase})', adjusted_running_bcs_acc, global_step=global_step)
                epoch_bcs_loss = update_dict(epoch_bcs_loss, running_bcs_loss)
                epoch_bcs_correct = update_dict(epoch_bcs_correct, running_bcs_correct)
                running_bcs_loss = {}
                running_bcs_correct = {}
            else:
                t_logger.log_histogram(f'running_sample_exited_at ({phase})', running_sample_exited_at, global_step=global_step)
                epoch_sample_exited_at = update_tensor(epoch_sample_exited_at, running_sample_exited_at)

            epoch_loss += running_loss
            epoch_correct += running_correct
            epoch_denom += running_denom
            running_loss = 0.0
            running_correct = 0.0
            running_denom = 0.0

            if (i + 1) == loader_size:
                tmp_loss = epoch_loss / epoch_denom
                tmp_acc = epoch_correct / epoch_denom
                global_losses = {f'epoch_loss/{phase}': round(tmp_loss, 4),
                                 f'epoch_acc/{phase}': round(tmp_acc, 4)}
                t_logger.log_scalar(f'epoch_loss/{phase}', global_losses[f'epoch_loss/{phase}'], epoch)
                t_logger.log_scalar(f'epoch_acc/{phase}', global_losses[f'epoch_acc/{phase}'], epoch)
                if phase == 'train':
                    adjusted_epoch_bcs_loss = adjust_dict(epoch_bcs_loss, epoch_denom, 'loss', 'epoch')
                    adjusted_epoch_bcs_acc = adjust_dict(epoch_bcs_correct, epoch_denom, 'acc', 'epoch')
                    t_logger.log_scalars(f'epoch_loss_per_layer ({phase})', adjusted_epoch_bcs_loss, global_step=epoch)
                    t_logger.log_scalars(f'epoch_acc_per_layer ({phase})', adjusted_epoch_bcs_acc, global_step=epoch)
                else:
                    t_logger.log_histogram(f'epoch_sample_exited_at ({phase})', epoch_sample_exited_at, global_step=epoch)

            if lr_scheduler is not None and phase == 'train':
                t_logger.log_scalar('lr_scheduler', lr_scheduler.get_last_lr()[0], global_step)

def manual_seed(random_seed):
    """
    Set the environment for reproducibility purposes.
    Args:
        random_seed (int): seed generator
    """
    if 'cuda' in device.type:
        torch.cuda.empty_cache()
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(random_seed)
        # torch.backends.cudnn.benchmark = False
    import numpy as np
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

EPOCHS = 10

manual_seed(random_seed=42)
for epoch in range(EPOCHS):
    model.train()
    run_epoch(epoch, phase='train')
    model.eval()
    with torch.no_grad():
        run_epoch(epoch, phase='test')
