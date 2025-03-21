import torch
import torch.nn as nn
import torch.nn.functional as F


# class Actor(nn.Module):
#     """
#     Réseau d'acteur pour actions continues.
#     Input : état (vectorisé ou features extraites)
#     Output : 3 actions continues (entre -1 et 1).
#     """
#     def __init__(self, state_dim=40, action_dim=3, hidden_size=256):
#         super().__init__()
#         self.fc1 = nn.Linear(state_dim, hidden_size)  # 40 -> 256
#         self.fc2 = nn.Linear(hidden_size, hidden_size)
#         self.fc_out = nn.Linear(hidden_size, action_dim)
#
#     def forward(self, x):
#         x = F.relu(self.fc1(x))  # x shape: (batch_size, 40) => (batch_size, 256)
#         x = F.relu(self.fc2(x))
#         return torch.tanh(self.fc_out(x))

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim=3, hidden_size=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    """
    Réseau critique Q(s,a).
    Input : concaténation de (état, action)
    Output : Q-value (scalaire).
    """
    def __init__(self, state_dim, action_dim=3, hidden_size=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        # Concatène état et action
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)
