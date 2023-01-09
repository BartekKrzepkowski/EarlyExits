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

def correct_metric(y_pred, y_true):
    _, predicted = torch.max(y_pred.data, -1)
    correct = (predicted == y_true).sum().item()
    return correct

def update_dict(d1, dd1):
    if d1 == {}:
        d1 = dd1
    else:
        for k in d1:
            d1[k] += dd1[k]
    return d1

def adjust_dict(d, nom, metric, scope):
    new_d = {}
    for k in d:
        new_d[f'{scope}_{metric}/layer_{k}_training'] = d[k] / nom
    return new_d
