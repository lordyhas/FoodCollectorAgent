import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from networks import ActorCNN, CriticCNN
from replay_buffer import ReplayBuffer

class DDPGAgent:
    def __init__(self,
                 in_channels=5,  # le sensor fait 5 canaux
                 action_dim=3,
                 actor_lr=1e-4,
                 critic_lr=1e-3,
                 gamma=0.99,
                 tau=1e-3,
                 buffer_size=int(1e5),
                 batch_size=64,
                 device="cpu"):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        self.device = torch.device(device)
        self.writer = SummaryWriter(log_dir="runs/ddpg_cnn")

        # Réseaux
        self.actor = ActorCNN(in_channels, action_dim).to(self.device)
        self.critic = CriticCNN(in_channels, action_dim).to(self.device)

        # Réseaux cibles
        self.target_actor = ActorCNN(in_channels, action_dim).to(self.device)
        self.target_critic = CriticCNN(in_channels, action_dim).to(self.device)

        # Copier les poids initiaux
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        # Optimiseurs
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Replay Buffer
        self.memory = ReplayBuffer(capacity=buffer_size, batch_size=batch_size)

        # Exploration (noise)
        self.noise_scale = 1.0
        self.noise_decay = 0.9999
        self.noise_min = 0.1

        self.learn_steps = 0  # compteur d'updates

    def act(self, state_img, add_noise=True):
        """
        state_img: np.array shape (5,40,40)
        Retourne action (3,) dans [-1,1].
        """
        # Passer sur GPU, batch=1
        state_t = torch.tensor(state_img, dtype=torch.float32).unsqueeze(0).to(self.device)
        # => (1,5,40,40)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_t)  # shape (1,3)
            action = action.cpu().numpy().flatten()
        self.actor.train()

        if add_noise:
            noise = np.random.normal(0, self.noise_scale, size=action.shape)
            action = np.clip(action + noise, -1, 1)

        return action

    def step(self, state_img, action, reward, next_state_img, done):
        """
        Enregistre la transition + potentiellement learn() si buffer suffisant.
        """
        self.memory.push(state_img, action, reward, next_state_img, done)
        # On réduit peu à peu le bruit
        self.noise_scale = max(self.noise_min, self.noise_scale*self.noise_decay)

        if len(self.memory) > self.batch_size:
            self.learn()

    def learn(self):
        self.learn_steps += 1
        state_imgs, actions, rewards, next_imgs, dones = self.memory.sample()

        # Convertir en tenseurs
        state_imgs_t = torch.tensor(state_imgs, dtype=torch.float32).to(self.device)
        # (batch_size,5,40,40)
        actions_t = torch.tensor(actions, dtype=torch.float32).to(self.device)
        # (batch_size,3)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_imgs_t = torch.tensor(next_imgs, dtype=torch.float32).to(self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        # 1) Update Critic
        # Q_target = r + gamma * Q_target(s', pi_target(s')) * (1-done)
        with torch.no_grad():
            next_actions = self.target_actor(next_imgs_t)
            q_target_next = self.target_critic(next_imgs_t, next_actions)
            q_target = rewards_t + self.gamma * q_target_next * (1 - dones_t)

        q_expected = self.critic(state_imgs_t, actions_t)
        critic_loss = F.mse_loss(q_expected, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 2) Update Actor
        predicted_actions = self.actor(state_imgs_t)
        actor_loss = -self.critic(state_imgs_t, predicted_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 3) Soft update
        self.soft_update(self.actor, self.target_actor)
        self.soft_update(self.critic, self.target_critic)

        # Log dans TensorBoard
        self.writer.add_scalar("Loss/Actor", actor_loss.item(), self.learn_steps)
        self.writer.add_scalar("Loss/Critic", critic_loss.item(), self.learn_steps)

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_( self.tau*local_param.data + (1.0 - self.tau)*target_param.data )
