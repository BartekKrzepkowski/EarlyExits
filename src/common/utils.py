import os
from datetime import datetime

import torch
from torch.nn import functional as F


def configure_optimizer(optim, backbone, head, lr_backbone, lr_head, weight_decay=1e-4, **optim_kwargs):
    alert_chunks = ['embeddings', 'LayerNorm', 'bias']
    no_decay = {pn for pn, p in backbone.named_parameters() if any(c in pn for c in alert_chunks)}
    optimizer_grouped_parameters = [
        {
            "params": [p for pn, p in backbone.named_parameters() if pn not in no_decay and p.requires_grad],
            "weight_decay": weight_decay,
            'lr': lr_backbone
        },
        {
            "params": [p for pn, p in backbone.named_parameters() if pn in no_decay and p.requires_grad],
            "weight_decay": 0.0,
            'lr': lr_backbone
        },
    ]
    if head is not None:
        no_decay = {pn for pn, p in head.named_parameters() if any(c in pn for c in alert_chunks)}
        optimizer_grouped_parameters2 = [
            {
                "params": [p for pn, p in head.named_parameters() if pn not in no_decay and p.requires_grad],
                "weight_decay": weight_decay,
                'lr': lr_head
            },
            {
                "params": [p for pn, p in head.named_parameters() if pn in no_decay and p.requires_grad],
                "weight_decay": 0.0,
                'lr': lr_head
            },
        ]
        optimizer_grouped_parameters += optimizer_grouped_parameters2

    optimizer = optim(optimizer_grouped_parameters, **optim_kwargs)
    return optimizer


def adjust_evaluators(d1, dd2, denom, scope, phase):
    for evaluator_key in dd2:
        eval_key = str(evaluator_key).split('/')
        eval_key = eval_key[0] if len(eval_key) == 1 else '/'.join(eval_key[:-1])
        eval_key = eval_key.split('_')
        eval_key = '_'.join(eval_key[1:]) if eval_key[0] in {'running', 'epoch'} else '_'.join(eval_key)
        d1[f'{scope}_{eval_key}/{phase}'] += dd2[evaluator_key] * denom
    return d1


def adjust_evaluators_pre_log(d1, denom, round_at=4):
    d2 = {}
    for k in d1:
        d2[k] = round(d1[k] / denom, round_at)
    return d2


def update_tensor(a, b):
    c = torch.cat([a, b])
    return c


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model


def create_paths(base_path, exp_name):
    date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_path = os.path.join(os.getcwd(), f'{base_path}/{exp_name}/{date}')
    save_path = f'{base_path}/checkpoints'
    return base_path, save_path


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
