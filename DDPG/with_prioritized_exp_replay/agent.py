import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt

from replay_buffer import PrioritizedReplayBuffer
from models import Actor, Critic
from utils import OUNoise
from torch.utils.tensorboard import SummaryWriter


class DDPGagent:
    def __init__(self,
                 state_dim,
                 action_dim=3,
                 hidden_size=256,
                 actor_lr=1e-4,
                 critic_lr=1e-3,
                 gamma=0.99,
                 tau=1e-2,
                 max_memory_size=50000,
                 device="cuda"):
        self.num_states = state_dim
        self.num_actions = action_dim
        self.gamma = gamma
        self.tau = tau
        self.device = torch.device(device)

        self.loss_writer = SummaryWriter(log_dir="runs/loss")
        self.rewards_writer = SummaryWriter(log_dir="runs/rewards")

        self.learn_steps = 0

        self.critic_losses = []
        self.actor_losses = []
        self.mean_critic_losses = []
        self.mean_actor_losses = []
        self.rewards = []
        self.mean_rewards = []

        self.noise_scale = 1.0
        self.noise_decay = 0.9999
        self.noise_min = 0.1

        # Actor, Critic
        self.actor = Actor(self.num_states, hidden_size, self.num_actions).to(self.device)
        self.actor_target = Actor(self.num_states, hidden_size, self.num_actions).to(self.device)
        self.critic = Critic(self.num_states + self.num_actions, hidden_size, 1).to(self.device)
        self.critic_target = Critic(self.num_states + self.num_actions, hidden_size, 1).to(self.device)

        # Copy initial weights
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        # Memory & OU noise
        self.memory = PrioritizedReplayBuffer(max_memory_size)
        self.noise = OUNoise(action_dim=self.num_actions)

        self.critic_criterion = nn.MSELoss()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

    def get_action(self, state, t=0, add_noise=True):
        """
        state: np.array shape (state_dim,)
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # => (1, state_dim)
        action = self.actor(state_t).detach().cpu().numpy()[0]           # => (action_dim,)
        if add_noise:
            # action = self.noise.get_action(action, t)
            noise = np.random.normal(0, self.noise_scale, size=action.shape)
            action = np.clip(action + noise, -1, 1)
        return action

    def update1(self, batch_size=128):
        self.learn_steps += 1

        if len(self.memory) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.FloatTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Critic update
        Qvals = self.critic(states_t, actions_t)
        next_actions = self.actor_target(next_states_t).detach()
        next_Q = self.critic_target(next_states_t, next_actions)
        Qprime = rewards_t + self.gamma*next_Q*(1 - dones_t)

        critic_loss = self.critic_criterion(Qvals, Qprime)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        policy_loss = -self.critic(states_t, self.actor(states_t)).mean()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        # Soft update
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data*(1.0 - self.tau))

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data*(1.0 - self.tau))

        # Log dans TensorBoard
        self.loss_writer.add_scalar("Loss/Critic", critic_loss.item(), self.learn_steps)
        self.loss_writer.add_scalar("Loss/Actor", policy_loss.item(), self.learn_steps)

        # Store the losses in lists
        self.critic_losses.append(critic_loss.item())
        self.actor_losses.append(policy_loss.item())

        # Store the mean losses in lists
        self.mean_critic_losses.append(np.mean(self.critic_losses))
        self.mean_actor_losses.append(np.mean(self.actor_losses))


    def update(self, batch_size=128, beta=0.4):
        if len(self.memory) < batch_size:
            return

        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(batch_size, beta)
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.FloatTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        weights_t = torch.FloatTensor(weights).unsqueeze(1).to(self.device)

        # Critic update avec IS weights
        Qvals = self.critic(states_t, actions_t)
        next_actions = self.actor_target(next_states_t).detach()
        next_Q = self.critic_target(next_states_t, next_actions)
        Qprime = rewards_t + self.gamma * next_Q * (1 - dones_t)

        # Erreur TD pondérée
        td_errors = Qvals - Qprime
        critic_loss = (weights_t * (td_errors ** 2)).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        policy_loss = -self.critic(states_t, self.actor(states_t)).mean()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        # Soft update
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        # Mise à jour des priorités avec la nouvelle erreur TD (ex. la norme L1)
        new_priorities = td_errors.abs().detach().cpu().numpy().flatten() + 1e-6
        self.memory.update_priorities(indices, new_priorities)

        # Log dans TensorBoard
        self.loss_writer.add_scalar("Loss/Critic", critic_loss.item(), self.learn_steps)
        self.loss_writer.add_scalar("Loss/Actor", policy_loss.item(), self.learn_steps)

        # Store the losses in lists
        self.critic_losses.append(critic_loss.item())
        self.actor_losses.append(policy_loss.item())

        # Store the mean losses in lists
        self.mean_critic_losses.append(np.mean(self.critic_losses))
        self.mean_actor_losses.append(np.mean(self.actor_losses))


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