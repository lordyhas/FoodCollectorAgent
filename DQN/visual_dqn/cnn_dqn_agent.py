import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque

# Vérifier si CUDA est disponible, sinon utiliser CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyperparamètres
BUFFER_SIZE = int(1e5)   # Taille du replay buffer
BATCH_SIZE = 64          # Taille du batch pour l'entraînement
GAMMA = 0.99             # Facteur de réduction
TAU = 1e-3               # Pour les mises à jour soft du réseau cible
LR = 5e-4                # Taux d'apprentissage
UPDATE_EVERY = 4         # Fréquence de mise à jour du réseau

class CNNQNetwork(nn.Module):
    """Réseau de neurones CNN pour l'approximation de la fonction Q."""
    
    def __init__(self, input_channels, action_size, seed, fc1_units=512, fc2_units=256):
        """Initialisation des paramètres et construction du modèle.
        
        Params
        ======
            input_channels (int): Nombre de canaux d'entrée (5 pour FoodCollector visuel)
            action_size (int): Dimension de l'espace d'action
            seed (int): Graine aléatoire
            fc1_units (int): Nombre de neurones dans la première couche fully connected
            fc2_units (int): Nombre de neurones dans la deuxième couche fully connected
        """
        super(CNNQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        # Couches convolutionnelles
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        # Couches de pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calcul de la taille après les couches de convolution et pooling
        # Image initiale: 40x40 => Après 3 couches de pooling 2x2: 5x5
        conv_output_size = 64 * 5 * 5
        
        # Couches fully connected
        self.fc1 = nn.Linear(conv_output_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        
        # Normalisation par batch (améliore la stabilité de l'entraînement)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Dropout (réduit l'overfitting)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, state):
        """Construction du réseau qui mappe les états sur les valeurs Q."""
        # Si l'entrée est de forme (batch_size, 5, 40, 40), on peut directement l'utiliser
        # Si l'entrée est de forme (5, 40, 40), on ajoute une dimension de batch
        if state.dim() == 3:
            state = state.unsqueeze(0)  # Ajouter dimension batch_size
        
        # Couches convolutionnelles avec ReLU, BatchNorm et MaxPooling
        x = self.pool(F.relu(self.bn1(self.conv1(state))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Aplatir la sortie pour les couches fully connected
        x = x.view(x.size(0), -1)
        
        # Couches fully connected avec ReLU et Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        
        # Couche de sortie (valeurs Q pour chaque action)
        return self.fc3(x)

class CNNDQNAgent():
    """Agent qui interagit avec et apprend de l'environnement."""
    
    def __init__(self, state_shape, action_size, seed):
        """Initialisation d'un agent.
        
        Params
        ======
            state_shape (tuple): Forme de l'état (5, 40, 40) pour FoodCollector visuel
            action_size (int): Dimension de l'espace d'action
            seed (int): Graine aléatoire
        """
        self.input_channels = state_shape[0]
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        # Réseaux Q 
        self.qnetwork_local = CNNQNetwork(self.input_channels, action_size, seed).to(device)
        self.qnetwork_target = CNNQNetwork(self.input_channels, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        
        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        
        # Initialisation du compteur de temps
        self.t_step = 0
        
        # Pour le suivi de la perte
        self.last_loss = None
    
    def step(self, state, action, reward, next_state, done):
        # Sauvegarder l'expérience dans le replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Apprendre toutes les UPDATE_EVERY étapes
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # Si suffisamment d'échantillons dans le replay memory, prendre un échantillon et apprendre
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)
    
    def act(self, state, eps=0.):
        """Retourne les actions pour un état donné suivant la politique actuelle.
        
        Params
        ======
            state (array_like): État actuel
            eps (float): Epsilon, pour la sélection epsilon-greedy
        
        Returns
        ======
            action (int): Action sélectionnée
        """
        # Convertir l'état en tensor
        state = torch.from_numpy(state).float().to(device)
        
        # Passer en mode évaluation
        self.qnetwork_local.eval()
        
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        
        # Revenir en mode entraînement
        self.qnetwork_local.train()
        
        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
    
    def learn(self, experiences, gamma):
        """Mise à jour des paramètres du modèle à partir d'un batch d'expériences.
        
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done)
            gamma (float): facteur de réduction
        """
        states, actions, rewards, next_states, dones = experiences
        
        # Obtenir les valeurs Q max pour les prochains états depuis le réseau cible
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        # Calculer les valeurs Q cibles pour l'état actuel
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        
        # Obtenir les valeurs Q attendues du réseau local
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        # Calculer la perte
        loss = F.mse_loss(Q_expected, Q_targets)
        
        # Sauvegarder la valeur de perte pour le suivi
        self.last_loss = loss.item()
        
        # Minimiser la perte
        self.optimizer.zero_grad()
        loss.backward()
        # Limiter la norme du gradient (stabilité)
        torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 1)
        self.optimizer.step()
        
        # ------------------- Mise à jour du réseau cible ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)
    
    def soft_update(self, local_model, target_model, tau):
        """Mise à jour soft des paramètres du modèle.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        
        Params
        ======
            local_model (PyTorch model): Poids source
            target_model (PyTorch model): Poids destination
            tau (float): Paramètre d'interpolation
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class ReplayBuffer:
    """Buffer à taille fixe pour stocker les expériences."""
    
    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialisation d'un ReplayBuffer.
        
        Params
        ======
            action_size (int): Dimension de l'espace d'action
            buffer_size (int): Taille maximale du buffer
            batch_size (int): Taille du mini-batch
            seed (int): Graine aléatoire
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Ajouter une nouvelle expérience au buffer."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Échantillonner aléatoirement un batch d'expériences du buffer."""
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.array([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.array([[e.action] for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.array([[e.reward] for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.array([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.array([[e.done] for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        """Retourne la taille actuelle du buffer."""
        return len(self.memory) 