import torch
from accelerate import Accelerator
from transformers import get_cosine_schedule_with_warmup

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

from trainer import EarlyExitTrainer
from pee_method import PEE
from resnets import ResNet18

EPOCHS = 5
BATCH_SIZE = 128
GRAD_ACCUM_STEPS = 128 // BATCH_SIZE

from datasets import get_cifar10
from torch.utils.data import DataLoader
train_set, val_set, test_set = get_cifar10()

train_loader = DataLoader(dataset=train_set, shuffle=True, batch_size=BATCH_SIZE, pin_memory=True, num_workers=8)
val_loader = DataLoader(dataset=val_set, shuffle=False, batch_size=BATCH_SIZE, pin_memory=True, num_workers=8)
test_loader = DataLoader(dataset=test_set, shuffle=False, batch_size=BATCH_SIZE, pin_memory=True, num_workers=8)

NUM_TRAINING_STEPS = (len(train_loader) // GRAD_ACCUM_STEPS) * EPOCHS

backbone_model = ResNet18(10).to(device)
ee_method = PEE(backbone_model, [64, 64, 128, 128, 256, 256, 512, 512], 0.2, device).to(device)

accelerator = Accelerator()
criterion = torch.nn.CrossEntropyLoss().to(device)
optim = torch.optim.AdamW(ee_method.parameters(), **{'lr': 0.05, 'weight_decay': 0.001})
lr_scheduler = None

train_loader, val_loader, test_loader, ee_method, optim, lr_scheduler = accelerator.prepare(
        train_loader, val_loader, test_loader, ee_method, optim, lr_scheduler)

loaders = {'train': train_loader, 'test': test_loader, 'val': val_loader}

args_trainer = {
    'ee_method': ee_method,
    'criterion': criterion,
    'optim': optim,
    'accelerator': accelerator,
    'lr_scheduler': lr_scheduler,
    'loaders': loaders,
    'device': device
}

trainer = EarlyExitTrainer(**args_trainer)

import collections
config_run_epoch = collections.namedtuple('RE', ['save_interval', 'grad_accum_steps', 'running_step_mult'])(110000,
                                                                                                       GRAD_ACCUM_STEPS,
                                                                                                       4)
params_run = {
    'epoch_start': 0,
    'epoch_end': EPOCHS,
    'exp_name': f'gpfee',
    'config_run_epoch': config_run_epoch,
    'random_seed': 42
}
trainer.run_exp(**params_run)
