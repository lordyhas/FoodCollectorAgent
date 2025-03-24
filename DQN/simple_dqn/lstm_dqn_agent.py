import numpy as np
import random
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Hyperparamètres
BUFFER_SIZE = int(1e5)  # Taille du replay buffer
BATCH_SIZE = 64         # Taille du mini-batch
GAMMA = 0.99            # Facteur d'actualisation
TAU = 1e-3              # Coefficient de mise à jour du réseau cible
LR = 5e-4               # Taux d'apprentissage
UPDATE_EVERY = 4        # Fréquence d'actualisation du réseau
SEQUENCE_LENGTH = 8     # Longueur de séquence pour le LSTM

# Déterminer si un GPU est disponible
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Utilisation de l'appareil: {device}")

class LSTMQNetwork(nn.Module):
    """Réseau de neurones avec LSTM pour approximer la fonction Q."""
    
    def __init__(self, state_size, action_size, seed=42, hidden_size=64, lstm_layers=1):
        """Initialise les paramètres et construit le modèle.
        
        Params
        ======
            state_size (int): Dimension de chaque état
            action_size (int): Dimension de chaque action
            seed (int): Graine aléatoire
            hidden_size (int): Taille des unités cachées LSTM
            lstm_layers (int): Nombre de couches LSTM
        """
        super(LSTMQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        
        # Couche d'entrée pour traiter l'état
        self.fc_input = nn.Linear(state_size, hidden_size)
        
        # Couche LSTM
        self.lstm = nn.LSTM(hidden_size, hidden_size, 
                          num_layers=lstm_layers, 
                          batch_first=True)
        
        # Couche de sortie
        self.fc_output = nn.Linear(hidden_size, action_size)
        
        # Déplacer le modèle sur le GPU si disponible
        self.to(device)
        
    def forward(self, state, hidden_state=None):
        """Construit un réseau qui mappe l'état -> valeurs d'action."""
        # Redimensionner l'entrée si nécessaire pour les séquences
        batch_size = state.size(0)
        seq_length = 1
        if len(state.shape) == 3:  # Entrée de séquence [batch, seq_len, state_size]
            seq_length = state.size(1)
            state = state.reshape(-1, self.state_size)  # [batch*seq_len, state_size]
        
        # Passer par la couche d'entrée
        x = F.relu(self.fc_input(state))
        
        # Redimensionner pour LSTM
        x = x.reshape(batch_size, seq_length, self.hidden_size)
        
        # Initialiser l'état caché si non fourni
        if hidden_state is None:
            h_0 = torch.zeros(self.lstm_layers, batch_size, 
                              self.hidden_size).to(device)
            c_0 = torch.zeros(self.lstm_layers, batch_size, 
                              self.hidden_size).to(device)
            hidden_state = (h_0, c_0)
        
        # Passer à travers LSTM
        x, h_n = self.lstm(x, hidden_state)
        
        # Utiliser la dernière sortie de séquence pour prédiction
        x = x[:, -1, :]
        
        # Couche de sortie
        q_values = self.fc_output(x)
        
        return q_values, h_n
    
    def init_hidden(self, batch_size):
        """Initialise un nouvel état caché."""
        h_0 = torch.zeros(self.lstm_layers, batch_size, 
                          self.hidden_size).to(device)
        c_0 = torch.zeros(self.lstm_layers, batch_size, 
                          self.hidden_size).to(device)
        return (h_0, c_0)


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
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def sample_sequence(self, sequence_length=SEQUENCE_LENGTH):
        """Échantillonne des séquences d'expériences pour l'apprentissage LSTM."""
        if len(self.memory) < sequence_length + 1:
            return None
        
        # Sélectionner des points de départ aléatoires pour extraire des séquences
        start_indices = random.sample(range(len(self.memory) - sequence_length), self.batch_size)
        
        # Préparer les listes pour stocker les séquences
        states_batch = []
        actions_batch = []
        rewards_batch = []
        next_states_batch = []
        dones_batch = []
        
        # Extraire les séquences
        for start_idx in start_indices:
            states_seq = []
            actions_seq = []
            rewards_seq = []
            next_states_seq = []
            dones_seq = []
            
            for i in range(sequence_length):
                exp = self.memory[start_idx + i]
                states_seq.append(exp.state)
                actions_seq.append(exp.action)
                rewards_seq.append(exp.reward)
                next_states_seq.append(exp.next_state)
                dones_seq.append(exp.done)
            
            # Ajouter les séquences au batch
            states_batch.append(np.array(states_seq))
            actions_batch.append(np.array(actions_seq))
            rewards_batch.append(np.array(rewards_seq))
            next_states_batch.append(np.array(next_states_seq))
            dones_batch.append(np.array(dones_seq))
        
        # Convertir en tenseurs PyTorch
        states = torch.tensor(np.array(states_batch), dtype=torch.float).to(device)
        actions = torch.tensor(np.array(actions_batch), dtype=torch.long).to(device)
        rewards = torch.tensor(np.array(rewards_batch), dtype=torch.float).to(device)
        next_states = torch.tensor(np.array(next_states_batch), dtype=torch.float).to(device)
        dones = torch.tensor(np.array(dones_batch), dtype=torch.uint8).float().to(device)
        
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Retourne la taille actuelle de la mémoire interne."""
        return len(self.memory)


class LSTMDQNAgent:
    """Agent implémentant l'algorithme Deep Q-Learning avec LSTM."""
    
    def __init__(self, state_size, action_size, seed=42, sequence_length=SEQUENCE_LENGTH):
        """Initialise un objet Agent.
        
        Params
        ======
            state_size (int): Dimension de chaque état
            action_size (int): Dimension de chaque action
            seed (int): Graine aléatoire
            sequence_length (int): Longueur de séquence pour le LSTM
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.sequence_length = sequence_length
        
        # Réseaux Q avec LSTM
        self.qnetwork_local = LSTMQNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = LSTMQNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        
        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialiser le compteur de pas pour la mise à jour
        self.t_step = 0
        
        # Tampon d'état pour les entrées séquentielles
        self.state_buffer = deque(maxlen=sequence_length)
        # État caché actuel
        self.hidden_state = None
        # Variable pour stocker la dernière valeur de perte
        self.last_loss = None
    
    def step(self, state, action, reward, next_state, done):
        # Sauvegarder l'expérience dans le replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Ajouter l'état au tampon
        self.state_buffer.append(state)
        
        # Mise à jour du compteur de pas
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        
        # Réinitialiser l'état caché si l'épisode est terminé
        if done:
            self.hidden_state = None
            self.state_buffer.clear()
        
        if len(self.memory) > BATCH_SIZE * self.sequence_length and self.t_step == 0:
            # Si assez d'échantillons disponibles dans la mémoire, apprendre
            experiences = self.memory.sample_sequence(self.sequence_length)
            if experiences is not None:
                self.learn(experiences, GAMMA)
    
    def act(self, state, eps=0.):
        """Retourne l'action à prendre en fonction de l'état actuel, suivant la politique actuelle.
        
        Params
        ======
            state (array_like): état actuel
            eps (float): epsilon, pour l'exploration
        """
        # Ajouter l'état au tampon
        self.state_buffer.append(state)
        
        # Si le tampon n'est pas assez rempli, agir aléatoirement
        if len(self.state_buffer) < self.sequence_length and random.random() <= eps:
            return random.choice(np.arange(self.action_size))
        
        # Préparer l'entrée pour le réseau
        if len(self.state_buffer) < self.sequence_length:
            # Padding si nécessaire
            padded_buffer = list(self.state_buffer)
            while len(padded_buffer) < self.sequence_length:
                padded_buffer.insert(0, np.zeros_like(state))
            states = np.array(padded_buffer)
        else:
            states = np.array(list(self.state_buffer))
        
        states = torch.from_numpy(states).float().unsqueeze(0).to(device)  # [1, seq_len, state_size]
        
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values, self.hidden_state = self.qnetwork_local(states, self.hidden_state)
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
            experiences (Tuple[torch.Tensor]): tuple de (s, a, r, s', done) où s et s' sont des séquences
            gamma (float): facteur de remise
        """
        states, actions, rewards, next_states, dones = experiences
        
        # Obtenir les valeurs Q pour les états actuels
        q_local, _ = self.qnetwork_local(states)
        # On utilise seulement la dernière prédiction de la séquence pour l'apprentissage
        q_local = q_local.gather(1, actions[:, -1].unsqueeze(1))
        
        # Obtenir les valeurs Q pour les états suivants
        with torch.no_grad():
            q_target_next, _ = self.qnetwork_target(next_states)
            q_target_next = q_target_next.detach().max(1)[0].unsqueeze(1)
        
        # Calculer les valeurs Q cibles
        # On utilise la dernière récompense et l'état terminal
        q_target = rewards[:, -1].unsqueeze(1) + (gamma * q_target_next * (1 - dones[:, -1].unsqueeze(1)))
        
        # Calculer la perte
        loss = F.mse_loss(q_local, q_target)
        
        # Enregistrer la valeur de perte
        self.last_loss = loss.item()
        
        # Minimiser la perte
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Mettre à jour le réseau cible
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
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def _sample_from_buffer(self):
        """Extraire aléatoirement un batch d'expériences de la mémoire de replay."""
        experiences = random.sample(self.memory.memory, k=BATCH_SIZE)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones) 