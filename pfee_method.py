import torch

from utils import get_block_output, get_laser

#dodaj do trainera
class PFEE(torch.nn.Module):
    def __init__(self, backbone_model, n_layers, feature_dim, conf_thr):
        super().__init__()
        self.conf_thr = conf_thr
        self.n_layers = n_layers
        self.w_denom = n_layers * (n_layers + 1) / 2

        self.backbone_model = backbone_model
        self.loss_ce = torch.nn.CrossEntropyLoss()

        self.branch_classifiers = torch.nn.ModuleList([torch.Linear(feature_dim, feature_dim) for _ in range(n_layers)])
        self.learners = torch.nn.ModuleList([torch.Linear(feature_dim, feature_dim) for _ in range(n_layers)])
        self.adaptive_balance = torch.Linear(feature_dim, 1)

        #kazda strategia konkatenacji wymaga innego rozmiaru wejścia
        self.reduction_layers = torch.nn.ModuleList([torch.Linear(feature_dim * (i + 1), feature_dim)
                                                     for i in range(n_layers)])
        self.inc_strategy = self.concatenation


    def concatenation(self, layers_states):
        concat_repr = torch.cat(layers_states, dim=-1)
        nb = len(layers_states) - 1
        inc_state = self.reduction_layers[nb](concat_repr)
        return inc_state

    def cosine_loss(self, y_true, y_pred):
        cosine_sim = y_true.T @ y_pred / (torch.linalg.norm(y_pred, dim=-1) * torch.linalg.norm(y_true, dim=-1))
        return 1 - cosine_sim

    # zmien indeksy z 1:n na 0:n-1
    def run_train(self, x_true, y_true):
        '''

        :param laser: Layers' Aggregated SEquential Representations
        :param y_true: true label
        :return:
        '''
        layers_states = []
        for i, repr in enumerate(laser):
            state = self.branch_classifiers[i](repr)
            layers_states.append(state)
        o_loss = ce_loss = .0
        for i in range(0, self.n-1):
            s_pi = self.inc_strategy(layers_states[: i+1])

            imit_states = []
            imit_loss = 0
            state_i = layers_states[i]
            for j in range(i+1, self.n):
                approx_state_j = self.learners[j](state_i)
                imit_states.append(approx_state_j)
                state_j = layers_states[j]
                sim_loss = self.cosine_loss(state_j, approx_state_j)
                imit_loss += sim_loss
            f_si = self.inc_strategy(imit_states)

            imit_loss /= (self.n - i)
            o_loss += imit_loss

            balance = torch.sigmoid(self.adaptive_balance(s_pi))
            z_i = balance * s_pi + (1 - balance) * f_si
            p_i = torch.softmax(z_i)
            ce_loss_i = self.loss_ce(p_i, y_true)

            ce_loss += i * ce_loss_i

        loss = o_loss / (self.n + 1) + ce_loss / self.w_denom
        return loss

    def run_val(self, x_true, y_true):
        layers_states = []
        h_repr = get_block_output(x_true, 0)
        for i in range(self.n):
            laser_i = get_laser_i(h_repr)
            state_i = self.branch_classifiers[i](laser_i)
            layers_states.append(state_i)
            s_pi = self.inc_strategy(layers_states[: i+1])

            imit_states = []
            for j in range(i+1, self.n):
                approx_state_j = self.learners[j](state_i)
                imit_states.append(approx_state_j)
            f_si = self.inc_strategy(imit_states)

            balance = torch.sigmoid(self.adaptive_balance(s_pi))
            z_i = balance * s_pi + (1 - balance) * f_si
            p_i = torch.softmax(z_i, dim=-1)
            ce_loss_i = self.loss_ce(p_i, y_true)

            if self.entropy(p_i) < self.conf_thr or (i+1) == self.n:
                break
            else:
                h_repr = get_block_output(h_repr, i+1)
        return ce_loss_i, torch.argmax(z_i, -1)

    def entropy(self, p):
        return - torch.sum(p * torch.log(p), dim=1)







