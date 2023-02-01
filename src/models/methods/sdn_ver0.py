import numpy as np
import torch
import torch.nn.functional as F

from utils import correct_metric
from models import StandardHead

class SDN(torch.nn.Module):
    def __init__(self, model, layers_dim, confidence_threshold, num_classes, is_model_frozen, device):
        super().__init__()
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.n_layers = len(layers_dim)
        self.is_model_frozen = is_model_frozen

        self.model = model.to(device)
        self.loss_ce = torch.nn.CrossEntropyLoss().to(device)

        self.branch_classifiers = torch.nn.ModuleList([
            StandardHead(in_channels, out_features=num_classes, pool_size=4).to(device)
            for in_channels in layers_dim])

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
            state_i = layers_states[i]
            ce_loss_i = self.loss_ce(state_i, y_true)
            ce_loss += ce_loss_i
            bcs_loss[i] = ce_loss_i.item() * denom # by nastepnie podzielić przez wspólny mianownik
            bcs_correct[i] = correct_metric(state_i, y_true)

        ce_loss_main_head = self.loss_ce(output_main_head, y_true)
        if not self.is_model_frozen:
            ce_loss += ce_loss_main_head
        bcs_loss[self.n_layers-1] = ce_loss_main_head.item() * denom # by nastepnie podzielić przez wspólny mianownik
        bcs_correct[self.n_layers-1] = correct_metric(output_main_head, y_true)
        loss = ce_loss
        correct = bcs_correct[self.n_layers-1] # w zbierac maxacc ze wszystkich warstw? albo (maxacc - acc z main heada)
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
            p_i = F.softmax(state_i, dim=-1)
            head_confidences_i = self.entropy(p_i)

            unresolved_samples_mask = sample_exited_at == -1
            exit_mask_global = unresolved_samples_mask.clone()
            exit_mask_local = (head_confidences_i <= self.confidence_threshold).cpu().squeeze(dim=-1)
            # exit_mask_global = torch.zeros_like(unresolved_samples_mask, dtype=torch.bool)
            exit_mask_global[unresolved_samples_mask] = exit_mask_local
            sample_exited_at[exit_mask_global] = head_idx
            exit_indices_global = exit_mask_global.nonzero().view(-1).tolist()
            if len(exit_mask_local.size()) == 0: # exit_mask_local # problem ze skalarem
                exit_mask_local = torch.tensor([exit_mask_local])
            exit_indices_local = exit_mask_local.nonzero().view(-1).tolist()
            assert len(exit_indices_global) == len(exit_indices_local), \
                f'exit_indices_global: {exit_indices_global} exit_indices_local: {exit_indices_local}'
            # assert torch.equal(unresolved_samples_mask, exit_mask_global), 'usm and emg tensors are not equal'
            for j, k in zip(exit_indices_global, exit_indices_local):
                sample_outputs[j] = state_i[k] # ????
            # head handled
            head_idx += 1
            # continue only if there are unresolved samples
            if (exit_mask_local).all():
                break
            # continue only with remaining sample subset
            repr_i = repr_i[~exit_mask_local]

        if not (exit_mask_local).all():
            repr_last, _ = fg.send(repr_i)
            _, output_main_head = fg.send(repr_i)
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


