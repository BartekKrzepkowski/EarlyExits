import numpy as np
import torch
import torch.nn.functional as F

from utils import correct_metric

class GPFEE(torch.nn.Module):
    def __init__(self, backbone_model, layers_dim, confidence_threshold, num_classes, device):
        super().__init__()
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.n_layers = len(layers_dim)
        self.w_denom = self.n_layers * (self.n_layers + 1) / 2

        self.backbone_model = backbone_model
        self.loss_ce = torch.nn.CrossEntropyLoss()

        self.branch_classifiers = torch.nn.ModuleList([torch.nn.Linear(dim, num_classes) for dim in layers_dim])
        self.learners = torch.nn.ModuleList([torch.nn.Linear(num_classes, num_classes) for _ in layers_dim])
        self.adaptive_balance = torch.nn.ModuleList([torch.nn.Linear(num_classes, 1) for _ in layers_dim])

        # czy potrzebne są dwie wersje?
        self.reduction_layers_past = torch.nn.ModuleList([torch.nn.Linear(num_classes * (i + 1), num_classes)
                                                     for i in range(self.n_layers)])
        self.reduction_layers_future = torch.nn.ModuleList([torch.nn.Linear(num_classes * (i + 1), num_classes)
                                                     for i in range(self.n_layers)])
        self.inc_strategy = self.concatenation
        self.cosine_loss = torch.nn.CosineEmbeddingLoss(margin=0.0, size_average=None, reduce=None, reduction='mean')

    def concatenation(self, layers_states, direction):
        concat_repr = torch.cat(layers_states, dim=-1)
        nb = len(layers_states) - 1
        if direction == 'past':
            inc_state = self.reduction_layers_past[nb](concat_repr)
        elif direction == 'future':
            inc_state = self.reduction_layers_future[nb](concat_repr)
        return inc_state

    # def cosine_loss(self, y_true, y_pred):
    #     cosine_sim = y_true.T @ y_pred / (torch.linalg.norm(y_pred, dim=-1) * torch.linalg.norm(y_true, dim=-1))
    #     return 1 - cosine_sim

    def run_train(self, x_true, y_true):
        '''

        :param laser: Layers' Aggregated SEquential Representations
        :param y_true: true label
        :return:
        '''
        layers_states = []
        bcs_loss = {}
        bcs_correct = {}
        fg, repr_i = self.backbone_model.forward_generator(x_true), None
        for i in range(self.n_layers):
            repr_i = fg.send(repr_i) # output of i-th layer
            adj_repr_i = self.backbone_model.adjust_repr(repr_i)
            state = self.branch_classifiers[i](adj_repr_i)
            layers_states.append(state)

        o_loss = ce_loss = .0
        for i in range(self.n_layers-1):
            s_pi = self.inc_strategy(layers_states[: i+1], 'past')

            imit_states = []
            imit_loss = 0
            state_i = layers_states[i]
            for j in range(i+1, self.n_layers):
                approx_state_j = self.learners[j](state_i)
                imit_states.append(approx_state_j)
                state_j = layers_states[j]
                # print(state_j.shape, approx_state_j.shape)
                sim_loss = self.cosine_loss(state_j, approx_state_j, torch.ones(y_true.shape).to(self.device))
                imit_loss += sim_loss

            f_si = self.inc_strategy(imit_states, 'future')

            imit_loss /= (self.n_layers - i - 1)
            o_loss += imit_loss

            balance = torch.sigmoid(self.adaptive_balance[i](s_pi))
            z_i = balance * s_pi + (1 - balance) * f_si
            ce_loss_i = self.loss_ce(z_i, y_true)

            ce_loss += (i+1) * ce_loss_i # upewnij się co do indeksowania

            bcs_loss[i] = ce_loss_i.item() * y_true.size(0) # by nastepnie podzielić przez wspólny mianownik
            bcs_correct[i] = correct_metric(z_i, y_true)

        # the last layer doesn't need imitation
        z_last = self.inc_strategy(layers_states, 'past')
        ce_loss_last = self.loss_ce(z_last, y_true)
        ce_loss += self.n_layers * ce_loss_last

        bcs_loss[self.n_layers-1] = ce_loss_last.item() * y_true.size(0) # by nastepnie podzielić przez wspólny mianownik
        bcs_correct[self.n_layers-1] = correct_metric(z_last, y_true)

        loss = o_loss / (self.n_layers - 1) + ce_loss / self.w_denom
        correct = bcs_correct[self.n_layers-1]

        return loss, correct, bcs_loss, bcs_correct

    def run_val(self, x_true, y_true):
        layers_states = []
        fg, repr_i = self.backbone_model.forward_generator(x_true), None
        sample_exited_at = torch.zeros(x_true.size(0), dtype=torch.int) - 1
        sample_outputs = [torch.Tensor() for _ in range(x_true.size(0))]
        head_idx = 0
        for i in range(self.n_layers-1):
            repr_i = fg.send(repr_i) # x
            adj_repr_i = self.backbone_model.adjust_repr(repr_i)
            state_i = self.branch_classifiers[i](adj_repr_i)# head_output
            layers_states.append(state_i)
            s_pi = self.inc_strategy(layers_states[: i+1], 'past')

            imit_states = []
            for j in range(i+1, self.n_layers):
                approx_state_j = self.learners[j](state_i)
                imit_states.append(approx_state_j)
            f_si = self.inc_strategy(imit_states, 'future')

            balance = torch.sigmoid(self.adaptive_balance[i](s_pi))
            z_i = balance * s_pi + (1 - balance) * f_si # head_output global
            p_i = F.softmax(z_i, dim=-1)

            head_confidences = self.entropy(p_i)
            # early exiting masks
            unresolved_samples_mask = sample_exited_at == -1
            exit_mask_local = (head_confidences <= self.confidence_threshold).cpu().squeeze(dim=-1)
            exit_mask_global = torch.zeros_like(unresolved_samples_mask, dtype=torch.bool)
            exit_mask_global[unresolved_samples_mask] = exit_mask_local
            # update sample head index array
            sample_exited_at[exit_mask_global] = head_idx
            # update sample return list
            exit_indices_global = exit_mask_global.nonzero().view(-1).tolist()
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
        # except StopIteration:
        #     exit_mask_global = unresolved_samples_mask
        #     # update sample head index array
        #     sample_exited_at[exit_mask_global] = head_idx
        #     # update sample return list
        #     exit_indices_global = exit_mask_global.nonzero().view(-1).tolist()
        #     for j, k in enumerate(exit_indices_global):
        #         sample_outputs[k] = z_i[j]
        if not (exit_mask_local).all():
            repr_last = fg.send(repr_i) # x
            adj_repr_last = self.backbone_model.adjust_repr(repr_last)
            state_last = self.branch_classifiers[-1](adj_repr_last)# head_output
            layers_states.append(state_last)
            z_last = self.inc_strategy(layers_states, 'past')

        outputs = torch.stack(sample_outputs).to(self.device)
        ce_loss = self.loss_ce(outputs, y_true)
        correct = correct_metric(outputs, y_true)
        return ce_loss, correct

    def entropy(self, p):
        return - torch.sum(p * torch.log(p), dim=-1) / np.log(p.size(-1))


