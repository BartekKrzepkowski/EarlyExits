from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from src.common.common import ACT_NAME_MAP
from src.models.metrics import acc_metric
from src.models.models import StandardHead


class EarlyExitABC(torch.nn.Module):
    def __init__(self, model, layers_dim, num_classes, config_ee):
        super().__init__()
        self.device = config_ee['device']
        self.confidence_threshold = config_ee['confidence_threshold']
        self.n_layers = len(layers_dim)
        self.is_model_frozen = config_ee['is_model_frozen']
        self.num_classes = num_classes

        self.model = model
        self.loss_ce = torch.nn.CrossEntropyLoss()

        hidden_dim = layers_dim[-1]
        self.branch_classifiers = torch.nn.ModuleList([
            torch.nn.Sequential(StandardHead(in_channels, out_features=hidden_dim, pool_size=4),
                                ACT_NAME_MAP[config_ee['act_name']]())
            for in_channels in layers_dim])


    def get_merged_layer_state(self, states, current_layer):
        pass

    def run_train(self, x_true, y_true):
        '''

        :param x_true: data input
        :param y_true: data label
        :return:
        '''
        past_states = []
        ics_loss = {}
        ics_correct = {}
        fg, repr_i = self.model.forward_generator(x_true), None
        for i in range(self.n_layers-1):
            repr_i, _ = fg.send(repr_i) # output of i-th layer of backbone
            state_i = self.branch_classifiers[i](repr_i)
            past_states.append(state_i)

        repr_i, _ = fg.send(repr_i)
        state_main_head, output_main_head = fg.send(repr_i)
        past_states.append(state_main_head)

        ce_loss = 0.0
        for i in range(self.n_layers-1):
            z_i = self.get_merged_layer_state(past_states[:i + 1], i)
            past_states[i] = past_states[i].detach()

            ce_loss_i = self.loss_ce(z_i, y_true)
            ce_loss += ce_loss_i * (i + 1)
            ics_loss[i] = ce_loss_i.item()
            ics_correct[i] = acc_metric(z_i, y_true)

        if not self.is_model_frozen:
            output_main_head = self.get_merged_layer_state(past_states, self.n_layers - 1)
            ce_loss_main_head = self.loss_ce(output_main_head, y_true)
            ce_loss += ce_loss_main_head * self.n_layers
        else:
            ce_loss_main_head = self.loss_ce(output_main_head, y_true)

        ics_loss[self.n_layers-1] = ce_loss_main_head.item()
        ics_correct[self.n_layers-1] = acc_metric(output_main_head, y_true)

        loss = ce_loss / self.loss_ce_reweight_normalizer
        correct = ics_correct[self.n_layers-1]

        evaluators = defaultdict(float)
        evaluators['overall_loss'] = loss.item()
        evaluators['overall_acc'] = correct

        return loss, evaluators, ics_loss, ics_correct

    def run_val(self, x_true, y_true):
        past_states = []
        fg, repr_i = self.model.forward_generator(x_true), None
        self.sample_exited_at = torch.zeros(x_true.size(0), dtype=torch.int) - 1
        self.sample_outputs = [torch.Tensor() for _ in range(x_true.size(0))]
        head_idx = 0
        for i in range(self.n_layers-1):
            repr_i, _ = fg.send(repr_i) # output of i-th layer of backbone
            state_i = self.branch_classifiers[i](repr_i) # head_output
            past_states.append(state_i)

            z_i = self.get_merged_layer_state(past_states, i)

            exit_mask_local = self.find_exit(z_i, head_idx, is_last=False)
            head_idx += 1
            # continue only if there are unresolved samples
            if (exit_mask_local).all():
                break
            # continue only with remaining sample subset
            repr_i = repr_i[~exit_mask_local]
            past_states = [state_i[~exit_mask_local] for state_i in past_states]

        if not (exit_mask_local).all():
            repr_last, _ = fg.send(repr_i)
            state_main_head, output_main_head = fg.send(repr_i)
            if not self.is_model_frozen:
                output_main_head = self.get_merged_layer_state(past_states + [state_main_head])
            p_main_head = F.softmax(output_main_head, dim=1)
            _ = self.find_exit(p_main_head, head_idx, is_last=True)

        outputs = torch.stack(self.sample_outputs).to(self.device)
        ce_loss = self.loss_ce(outputs, y_true)
        acc = acc_metric(outputs, y_true)

        evaluators = defaultdict(float)
        evaluators['ce_loss'] = ce_loss.item()
        evaluators['overall_acc'] = acc

        return evaluators, self.sample_exited_at

    def entropy(self, p):
        return - torch.sum(p * torch.log(p), dim=1) / np.log(p.size(-1))
    
    def find_exit(self, z_i, head_idx, is_last):
        p_i = F.softmax(z_i, dim=1)
        head_confidences_i = self.entropy(p_i)

        unresolved_samples_mask = self.sample_exited_at == -1
        exit_mask_global = unresolved_samples_mask.clone()
        exit_mask_local = (head_confidences_i <= self.confidence_threshold).cpu().squeeze(dim=-1)
        if not is_last:
            exit_mask_global[unresolved_samples_mask] = exit_mask_local
            self.sample_exited_at[exit_mask_global] = head_idx
            if len(exit_mask_local.size()) == 0: # exit_mask_local # problem ze skalarem
                exit_mask_local = torch.tensor([exit_mask_local])
            exit_indices_global = exit_mask_global.nonzero().view(-1).tolist()
            exit_indices_local = exit_mask_local.nonzero().view(-1).tolist()
            assert len(exit_indices_global) == len(exit_indices_local), \
                f'exit_indices_global: {exit_indices_global} exit_indices_local: {exit_indices_local}'
            for j, k in zip(exit_indices_global, exit_indices_local):
                self.sample_outputs[j] = z_i[k]
        else:
            exit_indices_global = exit_mask_global.nonzero().view(-1).tolist()
            for j, k in enumerate(exit_indices_global):
                self.sample_outputs[k] = z_i[j]

            exit_mask_global[unresolved_samples_mask] = exit_mask_local
            self.sample_exited_at[exit_mask_global] = head_idx

        return exit_mask_local





