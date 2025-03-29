import numpy as np
import random
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Hyperparamètres optimisés
BUFFER_SIZE = int(1e5)
BATCH_SIZE = 128  # Augmenté pour une meilleure stabilité
GAMMA = 0.99
TAU = 5e-3  # Augmenté pour des mises à jour plus fréquentes
LR = 3e-4  # Optimisé pour une meilleure convergence
UPDATE_EVERY = 4

# Déterminer si un GPU est disponible
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(f"Utilisation de l'appareil: {device}")


class QNetwork(nn.Module):
    """Réseau de neurones pour approximer la fonction Q."""

    def __init__(self, state_size, action_size):
        """Initialise les paramètres et construit le modèle.

        Params
        ======
            state_size (int): Dimension de chaque état
            action_size (int): Dimension de chaque action
        """
        super(QNetwork, self).__init__()

        # Architecture plus large et plus profonde
        self.fc1 = nn.Linear(state_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, action_size)

        # Initialisation des poids avec He
        self.apply(self._init_weights)

        # Dropout pour la régularisation
        self.dropout = nn.Dropout(0.1)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            module.bias.data.fill_(0.01)

    def forward(self, state):
        """Construit un réseau qui mappe l'état -> valeurs d'action."""
        # Normalisation de l'état
        x = state / state.abs().max() if state.abs().max() > 0 else state

        # Couches avec activation et normalisation
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        return self.fc4(x)


class DQNAgent:
    """Agent implémentant l'algorithme Deep Q-Learning."""

    def __init__(self, state_size, action_size, seed=None):
        """Initialise un objet Agent.

        Params
        ======
            state_size (int): Dimension de chaque état
            action_size (int): Dimension de chaque action
            seed (int): Graine aléatoire (optionnelle)
        """
        self.state_size = state_size
        self.action_size = action_size

        # Définir la graine aléatoire si fournie
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        # Réseaux Q
        self.qnetwork_local = QNetwork(state_size, action_size).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size).to(device)

        # Optimiseur
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = PrioritizedReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE)

        # Initialiser le pas de temps
        self.t_step = 0
        self.last_loss = 0.0

        # Paramètres pour l'exploration adaptative
        self.eps_min = 0.01
        self.eps_decay = 0.995
        self.eps = 1.0
        self.eps_threshold = 0.5  # Seuil pour ajuster epsilon
        self.performance_window = deque(maxlen=100)

    def step(self, state, action, reward, next_state, done):
        # Sauvegarder l'expérience dans la replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Apprentissage à chaque UPDATE_EVERY pas de temps
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0 and len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def act(self, state, eps=None):
        """Retourne l'action à prendre en fonction de l'état actuel, suivant la politique actuelle.

        Params
        ======
            state (array_like): état actuel
            eps (float): epsilon, pour l'exploration
        """
        if eps is None:
            eps = self.eps

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy avec exploration adaptative
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
        states, actions, rewards, next_states, dones, indices, weights = experiences

        # Double DQN
        next_actions = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
        Q_targets_next = self.qnetwork_target(next_states).detach().gather(1, next_actions)

        # Calcul des Q-values cibles avec clipping des rewards
        rewards = torch.clamp(rewards, -1, 1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Q-values courantes
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Calcul de la perte avec importance sampling
        td_errors = torch.abs(Q_expected - Q_targets).detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors)

        # Huber loss pour la robustesse
        loss = F.smooth_l1_loss(Q_expected * weights, Q_targets * weights)
        self.last_loss = loss.item()

        # Optimisation
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 1.0)
        self.optimizer.step()

        # Mise à jour du réseau cible
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

        # Mise à jour de l'exploration adaptative
        if len(self.performance_window) >= 100:
            avg_performance = np.mean(self.performance_window)
            if avg_performance > self.eps_threshold:
                self.eps = max(self.eps_min, self.eps * self.eps_decay)
            else:
                self.eps = min(1.0, self.eps / self.eps_decay)

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


class PrioritizedReplayBuffer:
    """Mémoire tampon à taille fixe pour stocker les expériences de l'agent."""

    def __init__(self, action_size, buffer_size, batch_size, alpha=0.6, beta=0.4):
        """Initialise un objet ReplayBuffer.

        Params
        ======
            action_size (int): Dimension de chaque action
            buffer_size (int): Taille maximale de la mémoire tampon
            batch_size (int): Taille de chaque échantillon d'entraînement
            alpha (float): Détermine l'importance de la priorité
            beta (float): Importance sampling weight
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        # Initialiser les priorités comme un tableau numpy dès le début
        self.priorities = np.ones(buffer_size, dtype=np.float32)  # Changement ici
        self.priorities_index = 0  # Pour suivre l'index courant
        self.alpha = alpha
        self.beta = beta
        self.eps = 1e-5

    def add(self, state, action, reward, next_state, done):
        """Ajoute une nouvelle expérience à la mémoire."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

        # Mettre à jour les priorités
        max_prio = np.max(self.priorities[:len(self.memory)]) if self.memory else 1.0
        if self.priorities_index >= len(self.priorities):
            self.priorities_index = 0
        self.priorities[self.priorities_index] = max_prio
        self.priorities_index += 1

    def sample(self):
        """Échantillonne aléatoirement un lot d'expériences de la mémoire."""
        if len(self.memory) == 0:
            return None

        # N'utiliser que les priorités des expériences existantes
        current_priorities = self.priorities[:len(self.memory)]
        probs = current_priorities ** self.alpha
        probs /= probs.sum()

        # Sélection des indices avec les probabilités calculées
        indices = np.random.choice(len(self.memory), self.batch_size, p=probs)

        # Calcul des weights pour l'importance sampling
        weights = (len(self.memory) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = torch.FloatTensor(weights).to(device)

        # Récupération des expériences
        experiences = [self.memory[idx] for idx in indices]

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones, indices, weights)

    def update_priorities(self, indices, td_errors):
        """Met à jour les priorités pour les indices donnés."""
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = float(error) + self.eps

    def __len__(self):
        """Retourne la taille actuelle de la mémoire interne."""
        return len(self.memory)