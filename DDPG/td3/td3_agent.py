import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from .networks import Actor, TwinCritics
from .replay_buffer import ReplayBuffer

class TD3Agent:
    def __init__(self,
                 state_dim,
                 action_dim=3,
                 actor_lr=1e-4,
                 critic_lr=1e-3,
                 gamma=0.99,
                 tau=1e-3,
                 buffer_size=int(1e5),
                 batch_size=64,
                 policy_delay=2,
                 policy_noise=0.2,
                 noise_clip=0.5,
                 device="cpu"):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.policy_delay = policy_delay
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.device = torch.device(device)

        # Actor et Critics
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critics = TwinCritics(state_dim, action_dim).to(self.device)

        # Target networks
        self.actor_target = Actor(state_dim, action_dim).to(self.device)
        self.critics_target = TwinCritics(state_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critics_target.load_state_dict(self.critics.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critics_optimizer = optim.Adam(self.critics.parameters(), lr=critic_lr)

        self.memory = ReplayBuffer(capacity=buffer_size, batch_size=batch_size)

        self.learn_step = 0  # pour différer la mise à jour actor

    def act(self, state, add_noise=True):
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_t).cpu().numpy().flatten()
        self.actor.train()
        if add_noise:
            # bruit gaussien
            noise = np.random.normal(0, 0.1, size=action.shape)
            action = np.clip(action + noise, -1, 1)
        return action

    def step(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
        if len(self.memory) > self.batch_size:
            self.learn()

    def learn(self):
        self.learn_step += 1

        states, actions, rewards, next_states, dones = self.memory.sample()
        states_t = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions_t = torch.tensor(actions, dtype=torch.float32).to(self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        # 1) Cible Q
        #   On ajoute un bruit sur l'action de l'acteur cible (target policy smoothing).
        with torch.no_grad():
            next_actions = self.actor_target(next_states_t)
            noise = (torch.randn_like(next_actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (next_actions + noise).clamp(-1, 1)

            q1_next, q2_next = self.critics_target(next_states_t, next_actions)
            q_next_min = torch.min(q1_next, q2_next)
            q_target = rewards_t + self.gamma * (1 - dones_t) * q_next_min

        # 2) Critic update
        q1, q2 = self.critics(states_t, actions_t)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.critics_optimizer.zero_grad()
        critic_loss.backward()
        self.critics_optimizer.step()

        # 3) Delay update de l'acteur (1 fois tous les policy_delay itérations)
        if self.learn_step % self.policy_delay == 0:
            # Actor update => max Q => min -Q => on prend Q1 par convention
            predicted_actions = self.actor(states_t)
            q1_actor, _ = self.critics(states_t, predicted_actions)
            actor_loss = -q1_actor.mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # 4) Soft update
            self.soft_update(self.actor, self.actor_target)
            self.soft_update(self.critics, self.critics_target)

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1 - self.tau) * target_param.data)
