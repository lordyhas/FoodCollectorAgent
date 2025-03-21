import numpy as np
import torch
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from ddpg.ddpg_agent import DDPGAgent

from ddpg.td3.td3_agent import TD3Agent


def main():
    # Config ML-Agents
    config_channel = EngineConfigurationChannel()
    env = UnityEnvironment(
        file_name="environment/FoodCollector.exe",
        side_channels=[config_channel],
        seed=0,
        no_graphics=True,
        worker_id=1
    )
    env.reset()

    behavior_name = list(env.behavior_specs)[0]
    spec = env.behavior_specs[behavior_name]

    # 1) Vérifier qu'il y a 3 actions continues
    assert spec.action_spec.continuous_size == 3, "On attend 3 actions continues dans FoodCollector"
    # On ignore la branche discrète => fixée à 0
    # 2) Récupérer la forme de l'observation. Suppose qu'il n'y a qu'un vector (state_dim).
    #    Si c'est un grid sensor => faudra un CNN. Simplifions en supposant state_dim = 10, par ex.
    #    Ici, just for example, on fait : state_dim = spec.observation_specs[0].shape[0]
    state_dim = spec.observation_specs[0].shape[0]

    # 3) Créer l'agent TD3
    agent = TD3Agent(state_dim=state_dim, action_dim=3, device="cuda" if torch.cuda.is_available() else "cpu")

    max_episodes = 500
    max_steps = 1000

    for episode in range(max_episodes):
        env.reset()
        decision_steps, terminal_steps = env.get_steps(behavior_name)

        done = False
        ep_reward = 0.0
        steps_count = 0

        # Récupérer un state initial
        if len(decision_steps) > 0:
            obs = decision_steps.obs[0][0]  # shape (state_dim,)
            state = obs
        else:
            state = np.zeros(state_dim, dtype=np.float32)
            done = True

        while not done and steps_count < max_steps:
            steps_count += 1

            # Action DDPG
            action_cont = agent.act(state, add_noise=True)
            # On ignore la discrète => fixons disc=0
            action_tuple = ActionTuple()
            action_tuple.add_continuous(np.array([action_cont], dtype=np.float32))  # shape (1,3)
            action_tuple.add_discrete(np.array([[0]], dtype=np.int32))  # fix disc=0

            env.set_actions(behavior_name, action_tuple)
            env.step()

            decision_steps, terminal_steps = env.get_steps(behavior_name)

            reward = 0.0
            next_state = state
            done = False

            if len(decision_steps) > 0:
                obs_next = decision_steps.obs[0][0]
                next_state = obs_next
                reward = decision_steps.reward[0]

            if len(terminal_steps) > 0:
                # Si agent dans terminal_steps => fin d'épisode
                for agent_id in terminal_steps:
                    reward = terminal_steps[agent_id].reward
                done = True

            ep_reward += reward

            # Entraîner l'agent
            agent.step(state, action_cont, reward, next_state, float(done))

            state = next_state

        print(f"[Episode {episode}] Reward={ep_reward:.2f}")

    env.close()
    print("DDPG Training done.")


if __name__ == "__main__":
    main()
