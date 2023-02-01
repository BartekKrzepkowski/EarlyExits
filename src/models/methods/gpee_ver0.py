import numpy as np
import torch
import torch.nn.functional as F

from utils import correct_metric
from models import StandardHead

class GPEE(torch.nn.Module):
    def __init__(self, model, layers_dim, confidence_threshold, num_classes, is_model_frozen, device):
        super().__init__()
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.n_layers = len(layers_dim)
        self.weight_normalizer = self.n_layers * (self.n_layers - 1) / 2 + (0 if  is_model_frozen else self.n_layers)
        self.is_model_frozen = is_model_frozen

        self.model = model.to(device)
        self.loss_ce = torch.nn.CrossEntropyLoss().to(device)

        self.branch_classifiers = torch.nn.ModuleList([
            StandardHead(in_channels, num_classes=num_classes, pool_size=4).to(device)
            for in_channels in layers_dim])

        self.reduction_layers_past = torch.nn.ModuleList([torch.nn.Linear(num_classes * (i + 1), num_classes).to(device)
                                                     for i in range(self.n_layers)])

        self.inc_strategy = self.concatenation

    def concatenation(self, layers_states):
        try:
            concat_repr = torch.cat(layers_states, dim=-1)
        except:
            for el in layers_states:
                print(el.shape)
        nb = len(layers_states) - 1
        inc_state = self.reduction_layers_past[nb](concat_repr)
        return inc_state

    def run_train(self, x_true, y_true):
        '''

        :param x_true: data input
        :param y_true: data label
        :return:
        '''
        layers_states = []
        bcs_loss = {}
        bcs_correct = {}
        fg, repr_i = self.model.forward_generator(x_true), None
        for i in range(self.n_layers-1): # nie dołączamy bc do ostatniej warstwy
            repr_i, _ = fg.send(repr_i) # output of i-th layer of backbone
            state_i = self.branch_classifiers[i](repr_i)
            layers_states.append(state_i)

        repr_i, _ = fg.send(repr_i)
        _, output_main_head = fg.send(repr_i)

        ce_loss = 0.0
        denom = y_true.size(0)
        for i in range(self.n_layers-1):
            z_i = self.inc_strategy(layers_states[: i+1])
            ce_loss_i = self.loss_ce(z_i, y_true)
            ce_loss += ce_loss_i * (i+1)
            bcs_loss[i] = ce_loss_i.item() * denom # by nastepnie podzielić przez wspólny mianownik
            bcs_correct[i] = correct_metric(state_i, y_true)

        if not self.is_model_frozen:
            output_main_head = self.inc_strategy(layers_states + [output_main_head])
            ce_loss_main_head = self.loss_ce(output_main_head, y_true)
            ce_loss += ce_loss_main_head + self.n_layers
        else:
            ce_loss_main_head = self.loss_ce(output_main_head, y_true)

        bcs_loss[self.n_layers-1] = ce_loss_main_head.item() * denom # by nastepnie podzielić przez wspólny mianownik
        bcs_correct[self.n_layers-1] = correct_metric(output_main_head, y_true)

        loss = ce_loss / self.weight_normalizer
        correct = bcs_correct[self.n_layers-1]
        return loss, correct, bcs_loss, bcs_correct

    def run_val(self, x_true, y_true):
        layers_states = []
        fg, repr_i = self.model.forward_generator(x_true), None
        sample_exited_at = torch.zeros(x_true.size(0), dtype=torch.int) - 1
        sample_outputs = [torch.Tensor() for _ in range(x_true.size(0))]
        head_idx = 0
        for i in range(self.n_layers-1):
            repr_i, _ = fg.send(repr_i) # output of i-th layer of backbone
            state_i = self.branch_classifiers[i](repr_i) # head_output
            layers_states.append(state_i)
            z_i = self.inc_strategy(layers_states[: i+1])
            p_i = F.softmax(z_i, dim=-1)
            head_confidences_i = self.entropy(p_i)

            unresolved_samples_mask = sample_exited_at == -1
            exit_mask_global = unresolved_samples_mask.clone()
            exit_mask_local = (head_confidences_i <= self.confidence_threshold).cpu().squeeze(dim=-1)
            exit_mask_global[unresolved_samples_mask] = exit_mask_local
            sample_exited_at[exit_mask_global] = head_idx
            exit_indices_global = exit_mask_global.nonzero().view(-1).tolist()
            if len(exit_mask_local.size()) == 0: # exit_mask_local # problem ze skalarem
                exit_mask_local = torch.tensor([exit_mask_local])
            exit_indices_local = exit_mask_local.nonzero().view(-1).tolist()
            assert len(exit_indices_global) == len(exit_indices_local), \
                f'exit_indices_global: {exit_indices_global} exit_indices_local: {exit_indices_local}'
            for j, k in zip(exit_indices_global, exit_indices_local):
                sample_outputs[j] = z_i[k] # ????
            # head handled
            head_idx += 1
            # continue only if there are unresolved samples
            if (exit_mask_local).all():
                break
            # continue only with remaining sample subset
            repr_i = repr_i[~exit_mask_local]
            layers_states = [state_i[~exit_mask_local] for state_i in layers_states]

        if not (exit_mask_local).all():
            repr_last, _ = fg.send(repr_i)
            _, output_main_head = fg.send(repr_i)
            if not self.is_model_frozen:
                output_main_head = self.inc_strategy(layers_states + [output_main_head])
            p_main_head = F.softmax(output_main_head, dim=-1)
            head_confidences_main_head = self.entropy(p_main_head)

            unresolved_samples_mask = sample_exited_at == -1
            exit_mask_global = unresolved_samples_mask.clone()
            exit_mask_local = (head_confidences_main_head <= self.confidence_threshold).cpu().squeeze(dim=-1)
            exit_indices_global = exit_mask_global.nonzero().view(-1).tolist()
            for j, k in enumerate(exit_indices_global):
                sample_outputs[k] = output_main_head[j]

            exit_mask_global[unresolved_samples_mask] = exit_mask_local
            sample_exited_at[exit_mask_global] = head_idx

        outputs = torch.stack(sample_outputs).to(self.device)
        ce_loss = self.loss_ce(outputs, y_true)
        correct = correct_metric(outputs, y_true)
        return ce_loss, correct, sample_exited_at

    def entropy(self, p):
        return - torch.sum(p * torch.log(p), dim=-1) / np.log(p.size(-1))


