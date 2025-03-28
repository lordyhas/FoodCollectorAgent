import torch
import torch.nn as nn
import torch.nn.functional as F

class TwinCritics(nn.Module):
    """
    Regroupe 2 critiques distincts.
    Input: (state, action)
    Output: Q1(s,a), Q2(s,a)
    """
    def __init__(self, state_dim, action_dim=3, hidden_size=256):
        super(TwinCritics, self).__init__()

        # Critic 1
        self.fc1_1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.fc1_2 = nn.Linear(hidden_size, hidden_size)
        self.q1_out = nn.Linear(hidden_size, 1)

        # Critic 2
        self.fc2_1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.fc2_2 = nn.Linear(hidden_size, hidden_size)
        self.q2_out = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)

        # Q1
        x1 = F.relu(self.fc1_1(x))
        x1 = F.relu(self.fc1_2(x1))
        q1 = self.q1_out(x1)

        # Q2
        x2 = F.relu(self.fc2_1(x))
        x2 = F.relu(self.fc2_2(x2))
        q2 = self.q2_out(x2)

        return q1, q2
