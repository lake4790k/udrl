import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):

    def __init__(self, state_size, action_size, seed):
        super().__init__()
        self.seed = torch.manual_seed(seed)

        self.ln1 = nn.Linear(state_size, 64)
        self.ln2 = nn.Linear(64, 16)
        self.ln3 = nn.Linear(16, action_size)

    def forward(self, state):
        h = F.relu(self.ln1(state))
        h = F.relu(self.ln2(h))
        return self.ln3(h)


class DuellingQNetwork(nn.Module):

    def __init__(self, state_size, action_size, seed):
        super().__init__()
        self.seed = torch.manual_seed(seed)

        self.ln1 = nn.Linear(state_size, 64)
        self.ln2 = nn.Linear(64, 16)

        self.adv = nn.Linear(16, action_size)
        self.v = nn.Linear(16, 1)

    def forward(self, state):
        h = F.relu(self.ln1(state))
        h = F.relu(self.ln2(h))

        a = self.adv(h)
        v = self.v(h)

        return v.expand_as(a) + a - a.mean(1, keepdim=True).expand_as(a)
