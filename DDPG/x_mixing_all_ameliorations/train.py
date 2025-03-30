import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from DDPG.config import MAX_EPISODES, MAX_STEPS
from DDPG.utils import CustomChannel
from agent import TD3Agent
from torch.utils.tensorboard import SummaryWriter


def main():
    config_channel = EngineConfigurationChannel()
    env = UnityEnvironment(
        file_name="../environment/GridFoodCollector.exe",
        side_channels=[config_channel, CustomChannel()],
        seed=0,
        no_graphics=True,
        worker_id=7
    )
    env.reset()

    behavior_name = list(env.behavior_specs)[0]
    spec = env.behavior_specs[behavior_name]

    # Vérification : on attend 3 actions continues.
    assert spec.action_spec.continuous_size == 3, "L'environnement doit avoir 3 actions continues"

    # On suppose que l'observation est donnée par un GridSensor de forme (5,40,40).
    obs_shape = spec.observation_specs[0].shape  # par exemple (5,40,40)
    state_dim = np.prod(obs_shape)  # 5*40*40 = 8000

    print("State dimension (flattened):", state_dim)

    agent = TD3Agent(state_dim=state_dim,
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
                     device="cuda",
                     use_param_noise=True)

    max_episodes = MAX_EPISODES
    max_steps = MAX_STEPS

    for episode in range(max_episodes):
        env.reset()
        decision_steps, terminal_steps = env.get_steps(behavior_name)

        done = False
        ep_reward = 0.0
        steps_count = 0

        if len(decision_steps) > 0:
            obs = decision_steps.obs[0][0]  # shape (5,40,40)
            state = obs.flatten()  # shape (8000,)
        else:
            state = np.zeros(state_dim, dtype=np.float32)
            done = True

        while not done and steps_count < max_steps:
            steps_count += 1

            # Obtenir l'action (avec OU noise et parameter noise)
            action = agent.get_action(state, t=steps_count, add_noise=True)

            # Construire ActionTuple : (1,3) pour la partie continue, ignore discrète
            at = ActionTuple()
            at.add_continuous(np.array([action], dtype=np.float32))
            at.add_discrete(np.array([[0]], dtype=np.int32))  # fixée à 0

            env.set_actions(behavior_name, at)
            env.step()

            decision_steps, terminal_steps = env.get_steps(behavior_name)
            reward = 0.0
            next_state = state

            if len(decision_steps) > 0:
                obs_next = decision_steps.obs[0][0]  # shape (5,40,40)
                next_state = obs_next.flatten()  # shape (8000,)
                reward = decision_steps.reward[0]

            if len(terminal_steps) > 0:
                # Fin d'épisode
                for agent_id in terminal_steps:
                    reward = terminal_steps[agent_id].reward
                done = True

            ep_reward += reward

            # Stocker la transition et mettre à jour l'agent
            agent.memory.push(state, action, reward, next_state, float(done))
            agent.update()

            state = next_state

        agent.rewards_writer.add_scalar("EpisodeReward", ep_reward, episode)
        print(f"Episode {episode}, Reward={ep_reward:.2f}, Steps={steps_count}")

        agent.rewards.append(ep_reward)
        agent.mean_rewards.append(np.mean(agent.rewards))

    env.close()
    agent.rewards_writer.close()
    agent.loss_writer.close()
    agent.plot_and_save_rewards(max_episodes)
    agent.plot_and_save_losses(max_episodes, max_steps)
    print("Training finished.")


if __name__ == "__main__":
    main()
