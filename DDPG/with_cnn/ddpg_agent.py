import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import datetime
from torch.utils.tensorboard import SummaryWriter
from collections import namedtuple, deque


from networks import ActorCNN, CriticCNN
from replay_buffer import ReplayBuffer

# Hyperparamètres optimisés
BUFFER_SIZE = int(1e5)


# Déterminer si un GPU est disponible
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DDPGAgent:
    def __init__(self,
                 in_channels=5,
                 action_dim=3,
                 actor_lr=1e-4,
                 critic_lr=1e-3,
                 gamma=0.99,
                 tau=1e-3,
                 buffer_size=int(1e5),
                 # batch_size=64,
                 batch_size=128,
                 device="cpu"):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        self.device = torch.device(device)
        self.loss_writer = SummaryWriter(log_dir="runs/loss")  # logs TensorBoard
        self.rewards_writer = SummaryWriter(log_dir="runs/rewards")  # logs TensorBoard

        # Réseaux
        self.actor = ActorCNN(in_channels, action_dim).to(self.device)
        self.critic = CriticCNN(in_channels, action_dim).to(self.device)

        # Réseaux cibles
        self.target_actor = ActorCNN(in_channels, action_dim).to(self.device)
        self.target_critic = CriticCNN(in_channels, action_dim).to(self.device)

        # Copie initiale
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        # Optimiseurs
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Replay Buffer
        self.memory = ReplayBuffer(capacity=buffer_size, batch_size=batch_size)
        # self.memory = PrioritizedReplayBuffer(action_dim, buffer_size, batch_size)


        # Param. exploration
        self.noise_scale = 1.0
        self.noise_decay = 0.9999
        self.noise_min = 0.1

        self.learn_steps = 0

        self.critic_losses = []
        self.actor_losses = []
        self.mean_critic_losses = []
        self.mean_actor_losses = []
        self.rewards = []
        self.mean_rewards = []


    def act(self, state_img, add_noise=True):
        """
        state_img: np.array (5,40,40)
        Retourne np.array shape (3,) dans [-1,1].
        """
        # batch=1
        state_t = torch.tensor(state_img, dtype=torch.float32).unsqueeze(0).to(self.device)
        # => shape (1,5,40,40)

        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_t)  # => (1,3)
            action = action.cpu().numpy().flatten()
        self.actor.train()

        if add_noise:
            noise = np.random.normal(0, self.noise_scale, size=action.shape)
            action = np.clip(action + noise, -1, 1)
        return action

    def step(self, state_img, action, reward, next_state_img, done):
        """
        Stocke la transition + lance un learn() si on a assez d'éléments.
        """
        self.memory.push(state_img, action, reward, next_state_img, done)
        # self.memory.add(state_img, action, reward, next_state_img, done)
        self.noise_scale = max(self.noise_min, self.noise_scale*self.noise_decay)

        if len(self.memory) > self.batch_size:
            # print(f"Learning at step {self.learn_steps}")
            self.learn()

    def learn(self):
        self.learn_steps += 1

        # Sample
        state_imgs, actions, rewards, next_imgs, dones = self.memory.sample()
        # state_imgs, actions, rewards, next_imgs, dones, indices, weights = self.memory.sample()
        state_imgs_t = torch.tensor(state_imgs, dtype=torch.float32).to(self.device)
        actions_t    = torch.tensor(actions, dtype=torch.float32).to(self.device)
        rewards_t    = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_imgs_t  = torch.tensor(next_imgs, dtype=torch.float32).to(self.device)
        dones_t      = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        # --- Critic update ---
        with torch.no_grad():
            next_actions = self.target_actor(next_imgs_t)
            q_next = self.target_critic(next_imgs_t, next_actions)
            q_target = rewards_t + self.gamma * q_next * (1 - dones_t)



        q_expected = self.critic(state_imgs_t, actions_t)

        critic_loss = F.smooth_l1_loss(q_expected, q_target)

        # Huber loss pour la robustesse
        # critic_loss = F.smooth_l1_loss(q_expected * weights, q_target * weights)


        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Actor update ---
        predicted_actions = self.actor(state_imgs_t)
        actor_loss = -self.critic(state_imgs_t, predicted_actions).mean()

        self.actor_optimizer.zero_grad()

        actor_loss.backward()
        self.actor_optimizer.step()

        # --- Soft Update ---
        self.soft_update(self.actor, self.target_actor)
        self.soft_update(self.critic, self.target_critic)

        # Log dans TensorBoard
        self.loss_writer.add_scalar("Loss/Critic", critic_loss.item(), self.learn_steps)
        self.loss_writer.add_scalar("Loss/Actor", actor_loss.item(), self.learn_steps)

        # Store the losses in lists
        self.critic_losses.append(critic_loss.item())
        self.actor_losses.append(actor_loss.item())

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

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                self.tau*local_param.data + (1.0-self.tau)*target_param.data
            )


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
