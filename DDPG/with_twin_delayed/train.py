import numpy as np
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from DDPG.config import MAX_EPISODES, MAX_STEPS
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from DDPG.utils import CustomChannel

from agent import TD3Agent  # Assurez-vous que le nom de fichier et de classe correspond


def main():
    config_channel = EngineConfigurationChannel()
    env = UnityEnvironment(
        file_name="../environment/GridFoodCollector.exe",
        side_channels=[config_channel, CustomChannel()],
        seed=0,
        no_graphics=True,
        worker_id=6
    )
    env.reset()
    behavior_name = list(env.behavior_specs)[0]
    spec = env.behavior_specs[behavior_name]

    # Vérification : on attend 3 actions continues
    assert spec.action_spec.continuous_size == 3, "L'environnement doit avoir 3 actions continues"
    # On ignore la branche discrète, fixée à 0.

    # Observation : on suppose un GridSensor de forme (5,40,40) → flatten en 8000
    obs_shape = spec.observation_specs[0].shape
    state_dim = 1
    for d in obs_shape:
        state_dim *= d
    print("State dim:", state_dim)

    agent = TD3Agent(state_dim=state_dim, action_dim=3, device="cuda" if torch.cuda.is_available() else "cpu")

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

            action = agent.get_action(state, add_noise=True)  # shape (3,)
            at = ActionTuple()
            at.add_continuous(np.array([action], dtype=np.float32))  # (1,3)
            at.add_discrete(np.array([[0]], dtype=np.int32))  # fixe disc=0

            env.set_actions(behavior_name, at)
            env.step()

            decision_steps, terminal_steps = env.get_steps(behavior_name)
            reward = 0.0
            next_state = state

            if len(decision_steps) > 0:
                obs_next = decision_steps.obs[0][0]
                next_state = obs_next.flatten()
                reward = decision_steps.reward[0]

            if len(terminal_steps) > 0:
                for agent_id in terminal_steps:
                    reward = terminal_steps[agent_id].reward
                done = True

            ep_reward += reward
            agent.memory.push(state, action, reward, next_state, float(done))
            agent.update()
            state = next_state

        print(f"Episode {episode}, Reward={ep_reward:.2f}, Steps={steps_count}")
        agent.rewards_writer.add_scalar("EpisodeReward", ep_reward, episode)
        agent.rewards.append(ep_reward)
        agent.mean_rewards.append(np.mean(agent.rewards))

    env.close()
    agent.rewards_writer.close()
    agent.loss_writer.close()
    agent.plot_and_save_rewards(max_episodes)
    agent.plot_and_save_losses(max_episodes, max_steps)


if __name__ == "__main__":
    main()
