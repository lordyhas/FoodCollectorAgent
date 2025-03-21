import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from networks import Actor, Critic
from replay_buffer import ReplayBuffer


class DDPGAgent:
    def __init__(self,
                 state_dim,
                 action_dim=3,
                 actor_lr=1e-4,
                 critic_lr=1e-3,
                 gamma=0.99,
                 tau=1e-3,
                 buffer_size=int(1e5),
                 batch_size=64,
                 device="cpu"):
        self.state_dim = state_dim  # Nouveau
        self.action_dim = action_dim  # Optionnel mais recommandé
        self.learn_steps = 0  # Compteur de pas d'entraînement
        self.writer = SummaryWriter(log_dir="runs/ddpg_foodcollector")

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = torch.device(device)

        # Réseaux
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)

        # Réseaux cibles
        self.target_actor = Actor(state_dim, action_dim).to(self.device)
        self.target_critic = Critic(state_dim, action_dim).to(self.device)

        # Copie initiale des poids
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        # Optimiseurs
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Replay buffer
        self.memory = ReplayBuffer(capacity=buffer_size, batch_size=batch_size)

        # Exploration param
        self.noise_scale = 1.0  # bruits initiaux
        self.noise_decay = 0.9999
        self.noise_min = 0.1

    def act(self, state, add_noise=True):
        """
        Sélection d'action : l'acteur produit une action (dans [-1,1]).
        On peut ajouter du bruit pour explorer.
        state : np.array shape (state_dim,)
        """
        # assert state.shape == (self.state_dim,), f"Expected shape {(self.state_dim,)}, got {state.shape}"
        assert state.shape == (self.state_dim,) or state.size == self.state_dim, \
            f"Expected shape ({self.state_dim},), got {state.shape}"

        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_t).cpu().numpy()
        self.actor.train()

        if add_noise:
            noise = np.random.normal(0, self.noise_scale, size=action.shape)
            action = np.clip(action + noise, -1, 1)
        return action

    def step(self, state, action, reward, next_state, done):
        """
        Enregistre la transition et déclenche l'apprentissage si buffer assez rempli.
        """
        self.memory.push(state, action, reward, next_state, done)
        # Décroît la noise scale
        self.noise_scale = max(self.noise_min, self.noise_scale * self.noise_decay)
        if len(self.memory) > self.batch_size:
            self.learn()

    def learn(self):
        """
        Echantillonne un batch, met à jour Critic et Actor, puis fait soft-update des cibles.
        """
        states, actions, rewards, next_states, dones = self.memory.sample()

        states_t = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions_t = torch.tensor(actions, dtype=torch.float32).to(self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        # 1) Mise à jour du Critic
        # Q_target = r + gamma * Q_target(s', pi_target(s')) * (1 - done)
        with torch.no_grad():
            next_actions = self.target_actor(next_states_t)
            q_target_next = self.target_critic(next_states_t, next_actions)
            q_target = rewards_t + (self.gamma * q_target_next * (1 - dones_t))

        q_expected = self.critic(states_t, actions_t)
        critic_loss = F.mse_loss(q_expected, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 2) Mise à jour de l'Actor
        # max a Q(s, actor(s)) => on calcule la loss = -Q(s, actor(s)).mean()
        predicted_actions = self.actor(states_t)
        actor_loss = -self.critic(states_t, predicted_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 3) Soft update des target networks
        self.soft_update(self.actor, self.target_actor)
        self.soft_update(self.critic, self.target_critic)

        # *** Logguer ces pertes dans TensorBoard
        self.writer.add_scalar("Loss/Critic", critic_loss.item(), self.learn_steps)
        self.writer.add_scalar("Loss/Actor", actor_loss.item(), self.learn_steps)

    def soft_update(self, local_model, target_model):
        """θ_target = τ*θ_local + (1 - τ)*θ_target"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )
