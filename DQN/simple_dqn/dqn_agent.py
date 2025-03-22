import numpy as np
import random
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Hyperparamètres
BUFFER_SIZE = int(1e5)  # Taille du replay buffer
BATCH_SIZE = 64  # Taille du mini-batch
GAMMA = 0.99  # Facteur d'actualisation
TAU = 1e-3  # Coefficient de mise à jour du réseau cible
LR = 5e-4  # Taux d'apprentissage
UPDATE_EVERY = 4  # Fréquence d'actualisation du réseau

# Déterminer si un GPU est disponible
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Utilisation de l'appareil: {device}")


class QNetwork(nn.Module):
    """Réseau de neurones pour approximer la fonction Q."""

    def __init__(self, state_size, action_size, seed=42, fc1_units=64, fc2_units=64):
        """Initialise les paramètres et construit le modèle.

        Params
        ======
            state_size (int): Dimension de chaque état
            action_size (int): Dimension de chaque action
            seed (int): Graine aléatoire
            fc1_units (int): Nombre de neurones dans la première couche cachée
            fc2_units (int): Nombre de neurones dans la deuxième couche cachée
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

        # Déplacer le modèle sur le GPU si disponible
        self.to(device)

    def forward(self, state):
        """Construit un réseau qui mappe l'état -> valeurs d'action."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    """Mémoire tampon à taille fixe pour stocker les expériences de l'agent."""

    def __init__(self, action_size, buffer_size, batch_size, seed=42):
        """Initialise un objet ReplayBuffer.

        Params
        ======
            action_size (int): Dimension de chaque action
            buffer_size (int): Taille maximale de la mémoire tampon
            batch_size (int): Taille de chaque échantillon d'entraînement
            seed (int): Graine aléatoire
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Ajoute une nouvelle expérience à la mémoire."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Échantillonne aléatoirement un lot d'expériences de la mémoire."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Retourne la taille actuelle de la mémoire interne."""
        return len(self.memory)


class DQNAgent:
    """Agent implémentant l'algorithme Deep Q-Learning."""

    def __init__(self, state_size, action_size, seed=42):
        """Initialise un objet Agent.

        Params
        ======
            state_size (int): Dimension de chaque état
            action_size (int): Dimension de chaque action
            seed (int): Graine aléatoire
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Réseaux Q
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialiser le compteur de pas pour la mise à jour
        self.t_step = 0
        # Variable pour stocker la dernière valeur de perte
        self.last_loss = None

    def step(self, state, action, reward, next_state, done):
        # Sauvegarder l'expérience dans le replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Mise à jour du compteur de pas
        self.t_step = (self.t_step + 1) % UPDATE_EVERY

        if len(self.memory) > BATCH_SIZE and self.t_step == 0:
            # Si assez d'échantillons disponibles dans la mémoire, apprendre
            experiences = self._sample_from_buffer()
            self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Retourne l'action à prendre en fonction de l'état actuel, suivant la politique actuelle.

        Params
        ======
            state (array_like): état actuel
            eps (float): epsilon, pour l'exploration
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Politique epsilon-greedy
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Mettre à jour les paramètres du réseau de valeurs en utilisant un batch d'expériences.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done)
            gamma (float): facteur de remise
        """
        states, actions, rewards, next_states, dones = experiences

        # Obtenir les valeurs Q maximales pour les états suivants à partir du réseau cible
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Calculer les valeurs Q cibles pour les états actuels
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Obtenir les valeurs Q attendues à partir du réseau local
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Calculer la perte
        loss = F.mse_loss(Q_expected, Q_targets)
        # Enregistrer la valeur de perte
        self.last_loss = loss.item()
        # Minimiser la perte
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- mettre à jour le réseau cible ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Mise à jour douce des paramètres du modèle cible.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): model with weights to copy from
            target_model (PyTorch model): model with weights to copy to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def _sample_from_buffer(self):
        """Extraire aléatoirement un batch d'expériences de la mémoire de replay."""
        experiences = random.sample(self.memory.memory, k=BATCH_SIZE)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)