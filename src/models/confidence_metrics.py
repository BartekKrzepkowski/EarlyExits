import numpy as np
import torch


def entropy_confidence(p, coef):
    entropy = - torch.sum(p * torch.log(p), dim=1) / np.log(p.size(-1))
    return entropy <= coef


def max_prob_confidence(p, coef):
    max_prob = torch.max(p, dim=1)[0]
    return max_prob >= coef
