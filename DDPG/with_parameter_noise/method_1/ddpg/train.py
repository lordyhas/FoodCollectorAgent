import uuid

import numpy as np
import sys

from mlagents_envs.side_channel import SideChannel
from torch.utils.tensorboard import SummaryWriter

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

from noisy_linear import NoisyLinear
from ddpg import DDPGagent

class CustomChannel(SideChannel):
    def __init__(self):
        super().__init__(uuid.UUID("a1d8f7b7-cec8-50f9-b78b-d3e165a78520"))
    def on_message_received(self, msg: bytes) -> None:
        pass


def main():
    # 1) Lance l'env Unity
    config_channel = EngineConfigurationChannel()
    env = UnityEnvironment(
        file_name="../environment/GridFoodCollector.exe",
        side_channels=[config_channel, CustomChannel()],
        seed=0,
        no_graphics=True,
        worker_id=2
    )
    env.reset()

    behavior_name = list(env.behavior_specs)[0]
    spec = env.behavior_specs[behavior_name]

    # On suppose 3 actions continues
    continuous_dim = spec.action_spec.continuous_size
    assert continuous_dim == 3, "FoodCollector (version 3 actions)"

    # Disons qu'on ignore la branche discrète => fix=0
    # On suppose un sensor (5,40,40) => flatten = 8000
    obs_shape = spec.observation_specs[0].shape  # e.g. (5,40,40)
    state_dim = 1
    for d in obs_shape:
        state_dim *= d

    print("State dim:", state_dim)
    agent = DDPGagent(
        state_dim=state_dim,
        action_dim=continuous_dim,
        hidden_size=256,
        actor_lr=1e-4,
        critic_lr=1e-3,
        gamma=0.99,
        tau=1e-2,
        max_memory_size=50000,
        device="cuda"
    )

    max_episodes = 50
    max_steps = 1000
    batch_size = 128

    all_rewards = []
    writer = agent.rewards_writer

    for episode in range(max_episodes):
        env.reset()
        decision_steps, terminal_steps = env.get_steps(behavior_name)

        # Observ init
        done = False
        episode_reward = 0.0
        steps_count = 0

        if len(decision_steps) > 0:
            obs = decision_steps.obs[0][0]  # shape(5,40,40)
            state = obs.flatten()           # => shape(8000,)
        else:
            state = np.zeros(state_dim, dtype=np.float32)
            done = True

        while not done and steps_count < max_steps:
            steps_count += 1
            # Action
            action = agent.get_action(state, t=steps_count, add_noise=True) # shape (3,)

            # ActionTuple
            at = ActionTuple()
            at.add_continuous(np.array([action], dtype=np.float32))  # => (1,3)
            at.add_discrete(np.array([[0]], dtype=np.int32))         # fix=0
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
                # L'agent est en terminal => end episode
                for agent_id in terminal_steps:
                    reward = terminal_steps[agent_id].reward
                done = True

            episode_reward += reward

            # store transition
            agent.memory.push(state, action, reward, next_state, float(done))

            # training step
            agent.update(batch_size)

            state = next_state

        all_rewards.append(episode_reward)
        writer.add_scalar("EpisodeReward", episode_reward, episode)
        agent.rewards.append(episode_reward)
        agent.mean_rewards.append(np.mean(agent.rewards))
        print(f"Episode: {episode}, reward={episode_reward:.2f}, steps={steps_count}")

    env.close()
    writer.close()
    print("Training finished. Rewards:")
    agent.plot_and_save_rewards(max_episodes)
    agent.plot_and_save_losses(max_episodes, max_steps)

if __name__=="__main__":
    main()
