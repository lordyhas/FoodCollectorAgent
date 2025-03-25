import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCNN(nn.Module):
    """
    Réseau Actor pour actions continues.
    Input : (batch,5,40,40). Output : (batch,3) dans [-1,1].
    """
    def __init__(self, in_channels=5, action_dim=3):
        super(ActorCNN, self).__init__()
        # Convolution layers
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # On aboutit à un volume (64,1,1) => flatten 64
        self.fc1 = nn.Linear(64, 256)
        self.fc2 = nn.Linear(256, action_dim)

    def forward(self, x):
        # x: (batch,5,40,40)
        x = F.relu(self.conv1(x))  # => (batch,32,9,9)
        x = F.relu(self.conv2(x))  # => (batch,64,3,3)
        x = F.relu(self.conv3(x))  # => (batch,64,1,1)
        x = x.view(x.size(0), -1)  # => (batch,64)
        x = F.relu(self.fc1(x))    # => (batch,256)
        # Sortie bornée dans [-1,1]
        return torch.tanh(self.fc2(x))

class CriticCNN(nn.Module):
    """
    Réseau Critic Q(s,a).
    Input s: (batch,5,40,40), a: (batch,3)
    Output: Q(s,a) scalaire => (batch,1)
    """
    def __init__(self, in_channels=5, action_dim=3):
        super(CriticCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Flatten => 64. Concat(action_dim=3) => 64+3=67
        self.fc1 = nn.Linear(64+action_dim, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, state_img, action):
        # state_img: (batch,5,40,40)
        s = F.relu(self.conv1(state_img))
        s = F.relu(self.conv2(s))
        s = F.relu(self.conv3(s))
        s = s.view(s.size(0), -1)  # => (batch,64)

        x = torch.cat([s, action], dim=1)  # => (batch,67)
        x = F.relu(self.fc1(x))            # => (batch,256)
        return self.fc2(x)                 # => (batch,1)
