import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCNN(nn.Module):
    """
    Actor pour actions continues, utilisant un CNN sur l'observation (5,40,40).
    Sortie : 3 actions dans [-1,1].
    """
    def __init__(self, in_channels=5, action_dim=3):
        super(ActorCNN, self).__init__()
        # Convolution layers
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Calculez la dimension de sortie après ces 3 conv
        # On peut faire un forward test ou calculer à la main :
        # Input (5,40,40)
        # conv1 : output ch=32, ~ (32, (40-8)/4+1=9, (40-8)/4+1=9)
        # conv2 : output ch=64, ~ (64, (9-4)/2+1=3, (9-4)/2+1=3)
        # conv3 : output ch=64, ~ (64, (3-3)/1+1=1, (3-3)/1+1=1)
        # => flatten = 64 * 1 * 1 = 64
        self.fc1 = nn.Linear(64, 256)
        self.fc2 = nn.Linear(256, action_dim)

    def forward(self, x):
        # x shape: (batch, 5, 40, 40)
        x = F.relu(self.conv1(x))  # => (batch,32,9,9)
        x = F.relu(self.conv2(x))  # => (batch,64,3,3)
        x = F.relu(self.conv3(x))  # => (batch,64,1,1)
        x = x.view(x.size(0), -1)  # => (batch,64)
        x = F.relu(self.fc1(x))    # => (batch,256)
        # On applique tanh pour borner l'action dans [-1,1]
        return torch.tanh(self.fc2(x))

class CriticCNN(nn.Module):
    """
    Critic Q(s,a) avec un CNN pour l'état (image).
    On extrait un embedding de l'état, puis on concatène l'action et on passe dans un MLP final.
    """
    def __init__(self, in_channels=5, action_dim=3):
        super(CriticCNN, self).__init__()
        # CNN sur l'état
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Après conv => 64 * 1 * 1 = 64
        # On concatène l'action (3) => total 64+3=67
        self.fc1 = nn.Linear(64 + action_dim, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, state, action):
        """
        state shape: (batch,5,40,40)
        action shape: (batch,3)
        """
        s = F.relu(self.conv1(state))
        s = F.relu(self.conv2(s))
        s = F.relu(self.conv3(s))
        s = s.view(s.size(0), -1)  # (batch,64)

        # Concaténer l'embedding d'état avec l'action
        x = torch.cat([s, action], dim=1)  # => (batch,64+3)
        x = F.relu(self.fc1(x))            # => (batch,256)
        return self.fc2(x)                 # => (batch,1)
