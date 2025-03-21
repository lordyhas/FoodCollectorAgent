import numpy as np
import time
import os
import logging
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.base_env import ActionTuple

# Configure logging to suppress specific warnings
logging.getLogger("mlagents").setLevel(logging.ERROR)
logging.getLogger("mlagents_envs").setLevel(logging.ERROR)
logging.getLogger("mlagents_envs.side_channel").setLevel(logging.ERROR)
logging.getLogger("mlagents_envs.environment").setLevel(logging.ERROR)


def run_random_agent(env_path, num_episodes=10, max_steps=1000, render=True):
    """
    Run an agent that takes random actions in the Unity environment.

    Parameters:
    -----------
    env_path: str
        Path to the Unity executable
    num_episodes: int
        Number of episodes to run
    max_steps: int
        Maximum steps per episode
    render: bool
        If True, runs in real-time for visualization
    """
    # Configure engine channel
    channel = EngineConfigurationChannel()

    # Initialize Unity environment
    print(f"Starting Unity environment: {env_path}")
    env = UnityEnvironment(file_name=env_path, side_channels=[channel], no_graphics=not render)

    # Set time scale based on render mode
    if render:
        channel.set_configuration_parameters(time_scale=1.0, width=1280, height=720)
    else:
        channel.set_configuration_parameters(time_scale=2.0)

    # Reset environment
    env.reset()

    # Get default behavior name
    behavior_name = list(env.behavior_specs)[0]
    print(f"Behavior name: {behavior_name}")

    # Get behavior specifications
    spec = env.behavior_specs[behavior_name]

    # Determine action space details
    if spec.action_spec.continuous_size > 0:
        print(f"Continuous actions: {spec.action_spec.continuous_size}")
        print(f"Discrete actions: {spec.action_spec.discrete_branches}")
        has_continuous = True
    else:
        print(f"Discrete actions only: {spec.action_spec.discrete_branches}")
        has_continuous = False

    # Scores tracker
    scores = []

    # Run episodes
    for i_episode in range(1, num_episodes + 1):
        env.reset()

        # Get initial state
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        agent_ids = decision_steps.agent_id
        num_agents = len(agent_ids)

        if num_agents == 0:
            print("No agents available. Check Unity environment.")
            break

        print(f"Episode {i_episode} | Number of agents: {num_agents}")
        score = 0

        # Run steps in episode
        for t in range(max_steps):
            # Take random actions for all agents
            if has_continuous:
                # Handle continuous actions
                continuous_actions = np.random.uniform(-1, 1, (num_agents, spec.action_spec.continuous_size))
                action_tuple = ActionTuple()
                action_tuple.add_continuous(continuous_actions)

                # Add discrete actions if needed
                if len(spec.action_spec.discrete_branches) > 0:
                    discrete_actions = np.array([
                        [np.random.randint(0, branch) for branch in spec.action_spec.discrete_branches]
                        for _ in range(num_agents)
                    ], dtype=np.int32)
                    action_tuple.add_discrete(discrete_actions)
            else:
                # Handle discrete actions only
                discrete_actions = np.array([
                    [np.random.randint(0, branch) for branch in spec.action_spec.discrete_branches]
                    for _ in range(num_agents)
                ], dtype=np.int32)
                action_tuple = ActionTuple()
                action_tuple.add_discrete(discrete_actions)

            # Execute action
            env.set_actions(behavior_name, action_tuple)
            env.step()

            # For visualization purposes
            if render and t % 5 == 0:
                time.sleep(0.01)

            # Get new state and reward
            decision_steps, terminal_steps = env.get_steps(behavior_name)

            # Collect rewards
            step_reward = 0
            for agent_id in decision_steps:
                step_reward += decision_steps[agent_id].reward
            for agent_id in terminal_steps:
                step_reward += terminal_steps[agent_id].reward

            score += step_reward

            # Check if episode is done (all agents in terminal state)
            if len(terminal_steps) == num_agents:
                break

        scores.append(score)
        print(f"Episode {i_episode}/{num_episodes} | Score: {score:.2f}")

    # Calculate and display average score
    mean_score = np.mean(scores)
    print(f"\nAverage score over {num_episodes} episodes: {mean_score:.2f}")

    # Close environment
    env.close()

    return scores


if __name__ == "__main__":
    # Path to Unity executable - update this to your executable path
    env_path = "../FoodCollector1Agent_Win64/FoodCollector.exe"

    # Run random agent
    run_random_agent(env_path, num_episodes=5, render=True)