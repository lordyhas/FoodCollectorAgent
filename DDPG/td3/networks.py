import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Twin Critic Network ---
class TwinCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(TwinCritic, self).__init__()
        # Critic 1
        self.fc1_1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.fc1_2 = nn.Linear(hidden_size, hidden_size)
        self.q1_out = nn.Linear(hidden_size, 1)
        # Critic 2
        self.fc2_1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.fc2_2 = nn.Linear(hidden_size, hidden_size)
        self.q2_out = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        # Concaténer état et action
        xu = torch.cat([state, action], dim=1)
        # Critic 1
        x1 = F.relu(self.fc1_1(xu))
        x1 = F.relu(self.fc1_2(x1))
        q1 = self.q1_out(x1)
        # Critic 2
        x2 = F.relu(self.fc2_1(xu))
        x2 = F.relu(self.fc2_2(x2))
        q2 = self.q2_out(x2)
        return q1, q2


# --- Actor Network ---
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # Utiliser tanh pour borner l'action dans [-1,1]
        x = torch.tanh(self.fc3(x))
        return x
