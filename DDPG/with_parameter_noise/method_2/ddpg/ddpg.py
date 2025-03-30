import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt

from models import Actor, Critic
from utils import Memory, OUNoise
from torch.utils.tensorboard import SummaryWriter


class DDPGagent:
    def __init__(self, state_dim, action_dim=3, hidden_size=256, actor_lr=1e-4, critic_lr=1e-3,
                 gamma=0.99, tau=1e-2, max_memory_size=50000, device="cpu", param_noise_scale=0.1):
        self.num_states = state_dim
        self.num_actions = action_dim
        self.gamma = gamma
        self.tau = tau
        self.device = torch.device(device)
        self.param_noise_scale = param_noise_scale

        # Réseaux
        self.actor = Actor(self.num_states, hidden_size, self.num_actions).to(self.device)
        self.actor_target = Actor(self.num_states, hidden_size, self.num_actions).to(self.device)
        self.critic = Critic(self.num_states + self.num_actions, hidden_size, 1).to(self.device)
        self.critic_target = Critic(self.num_states + self.num_actions, hidden_size, 1).to(self.device)

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        self.memory = Memory(max_memory_size)
        self.critic_criterion = nn.MSELoss()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Exploration par OU Noise (optionnel) si besoin
        self.ou_noise = OUNoise(action_dim=self.num_actions)

        # Pour loguer avec TensorBoard
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

        # Sauvegarde des paramètres originaux pour le parameter noise
        self.original_actor_params = [p.clone() for p in self.actor.parameters()]

    def perturb_actor_parameters(self):
        """
        Ajoute un bruit gaussien aux paramètres de l'acteur pour le parameter noise.
        """
        for orig_param, param in zip(self.original_actor_params, self.actor.parameters()):
            noise = torch.randn_like(param) * self.param_noise_scale
            param.data.copy_(orig_param.data + noise)

    def reset_actor_parameters(self):
        """
        Réinitialise les paramètres de l'acteur aux valeurs originelles.
        """
        for orig_param, param in zip(self.original_actor_params, self.actor.parameters()):
            param.data.copy_(orig_param.data)

    def get_action(self, state, t=0, add_noise=True, use_param_noise=False):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if use_param_noise:
            # Appliquer le parameter noise en début d'épisode
            self.perturb_actor_parameters()
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_t).cpu().numpy()[0]
        self.actor.train()
        if use_param_noise:
            # Réinitialiser pour la cohérence à la fin du step (ou épisode)
            self.reset_actor_parameters()
        if add_noise and not use_param_noise:
            # Optionnellement, ajouter OU noise
            # action = self.ou_noise.get_action(action, t=t)
            noise = np.random.normal(0, self.noise_scale, size=action.shape)
            action = np.clip(action + noise, -1, 1)
        return action

    def update(self, batch_size):
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
        Qprime = rewards_t + self.gamma * next_Q * (1 - dones_t)
        critic_loss = self.critic_criterion(Qvals, Qprime)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        policy_loss = -self.critic(states_t, self.actor(states_t)).mean()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        self.learn_steps += 1

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

        plt.savefig(f"figures_results/{current_date}_ddpg_cnn_loss_curves_{max_episodes}_{max_steps}.png")
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
        plt.savefig(f"figures_results/{current_date}_ddpg_cnn_rewards_curve_{max_episodes}.png")
        # plt.show()
