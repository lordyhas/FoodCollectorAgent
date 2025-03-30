import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from replay_buffer import ReplayBuffer
from networks import Actor, TwinCritic


# --- TD3 Agent ---
class TD3Agent:
    def __init__(self,
                 state_dim,
                 action_dim=3,
                 hidden_size=256,
                 actor_lr=1e-4,
                 critic_lr=1e-3,
                 gamma=0.99,
                 tau=1e-3,
                 buffer_size=int(1e5),
                 batch_size=128,
                 policy_noise=0.2,
                 noise_clip=0.5,
                 policy_delay=2,
                 device="cuda"):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.device = torch.device(device)

        # Réseau Actor
        self.actor = Actor(state_dim, action_dim, hidden_size).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, hidden_size).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # Réseau Critic Twin
        self.critics = TwinCritic(state_dim, action_dim, hidden_size).to(self.device)
        self.critics_target = TwinCritic(state_dim, action_dim, hidden_size).to(self.device)
        self.critics_target.load_state_dict(self.critics.state_dict())

        # Optimiseurs
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critics_optimizer = optim.Adam(self.critics.parameters(), lr=critic_lr)

        # Replay Buffer
        self.memory = ReplayBuffer(buffer_size)
        self.learn_steps = 0

        # Pour logging TensorBoard
        self.loss_writer = SummaryWriter(log_dir="runs/loss")
        self.rewards_writer = SummaryWriter(log_dir="runs/rewards")

        self.critic_losses = []
        self.actor_losses = []
        self.mean_critic_losses = []
        self.mean_actor_losses = []
        self.rewards = []
        self.mean_rewards = []

    def get_action(self, state, add_noise=True):
        """
        state: np.array shape (state_dim,) (ex: 8000 si flatten de (5,40,40))
        Retourne: np.array shape (action_dim,) dans [-1,1]
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_t).cpu().numpy()[0]
        self.actor.train()
        if add_noise:
            noise = np.random.normal(0, self.policy_noise, size=action.shape)
            action = np.clip(action + noise, -1, 1)
        return action

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        self.learn_steps += 1
        states, actions, rewards, next_states, dones = self.memory.sample()

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.FloatTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # --- Target Policy Smoothing ---
        with torch.no_grad():
            noise = (torch.randn_like(actions_t) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = self.actor_target(next_states_t)
            next_actions = (next_actions + noise).clamp(-1, 1)
            # Calculer Q-values avec les Twin Critics cibles
            q1_target, q2_target = self.critics_target(next_states_t, next_actions)
            q_target_next = torch.min(q1_target, q2_target)
            q_target = rewards_t + self.gamma * q_target_next * (1 - dones_t)

        # --- Critic Update ---
        q1, q2 = self.critics(states_t, actions_t)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critics_optimizer.zero_grad()
        critic_loss.backward()
        self.critics_optimizer.step()

        # --- Delayed Actor Update ---
        if self.learn_steps % self.policy_delay == 0:
            actor_loss = -self.critics(states_t, self.actor(states_t))[0].mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # --- Soft Update ---
            self.soft_update(self.actor, self.actor_target)
            self.soft_update(self.critics, self.critics_target)

            # Log actor loss
            self.loss_writer.add_scalar("Loss/Actor", actor_loss.item(), self.learn_steps)
            self.actor_losses.append(actor_loss.item())
            self.mean_actor_losses.append(np.mean(self.actor_losses))


        # Log critic loss
        self.loss_writer.add_scalar("Loss/Critic", critic_loss.item(), self.learn_steps)
        self.critic_losses.append(critic_loss.item())
        self.mean_critic_losses.append(np.mean(self.critic_losses))


    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1 - self.tau) * target_param.data)

    def plot_and_save_losses(self, max_episodes, max_steps):
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # Plot Critic Loss
        axs[0].plot(self.critic_losses, label='Critic Loss')
        # critic_mean = np.mean(self.critic_losses)
        axs[0].plot(self.mean_critic_losses, color='red', linestyle='--', label=f'Mean Critic Loss')
        axs[0].set_xlabel('Learning Steps')
        axs[0].set_ylabel('Loss')
        axs[0].legend()
        axs[0].set_title('Critic Loss Curve')

        # Plot Actor Loss
        axs[1].plot(self.actor_losses, label='Actor Loss')
        # actor_mean = np.mean(self.actor_losses)
        axs[1].plot(self.mean_actor_losses, color='red', linestyle='--', label=f'Mean Actor Loss')
        axs[1].set_xlabel('Learning Steps')
        axs[1].set_ylabel('Loss')
        axs[1].legend()
        axs[1].set_title('Actor Loss Curve')

        plt.tight_layout()

        current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        plt.savefig(f"figures_results/{current_date}_ddpg_loss_curves_{max_episodes}_{max_steps}.png")
        # plt.show()

    def plot_and_save_rewards(self, max_episodes):
        plt.figure()
        plt.plot(self.rewards, label='Episode Reward')
        plt.plot(self.mean_rewards, color='red', linestyle='--', label=f'Mean Actor Loss')

        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
        plt.title('Rewards Curve')

        current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(f"figures_results/{current_date}_ddpg_rewards_curve_{max_episodes}.png")
        # plt.show()