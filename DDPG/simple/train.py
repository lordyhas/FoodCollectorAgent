import uuid

import numpy as np
import torch
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel import SideChannel
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from ddpg_agent import DDPGAgent

class CustomChannel(SideChannel):
    def __init__(self):
        super().__init__(uuid.UUID("a1d8f7b7-cec8-50f9-b78b-d3e165a78520"))

    def on_message_received(self, msg: bytes) -> None:
        pass  # Ignorez ou traitez les messages

def main():
    # Config ML-Agents
    config_channel = EngineConfigurationChannel()
    try:
        env = UnityEnvironment(
            file_name="environment/GridFoodCollector.exe",
            side_channels=[EngineConfigurationChannel(), CustomChannel()],
            seed=0,
            no_graphics=True,
            worker_id=1
        )
        env.reset()

        behavior_name = list(env.behavior_specs)[0]
        spec = env.behavior_specs[behavior_name]

        print("Observation specs details:")
        for i, obs_spec in enumerate(spec.observation_specs):
            print(f"Obs {i}: shape={obs_spec.shape}, type={obs_spec.observation_type}")

        # 1) Vérifier qu'il y a 3 actions continues
        assert spec.action_spec.continuous_size == 3, "On attend 3 actions continues dans FoodCollector"
        # On ignore la branche discrète => fixée à 0
        # 2) Récupérer la forme de l'observation. Suppose qu'il n'y a qu'un vector (state_dim).
        #    Si c'est un grid sensor => faudra un CNN. Simplifions en supposant state_dim = 10, par ex.
        #    Ici, just for example, on fait : state_dim = spec.observation_specs[0].shape[0]
        # state_dim = spec.observation_specs[0].shape[0]
        state_dim = np.prod(spec.observation_specs[0].shape)

        # 3) Créer l'agent DDPG
        agent = DDPGAgent(state_dim=state_dim, action_dim=3, device="cuda" if torch.cuda.is_available() else "cpu")

        print(f"State dim: {state_dim}")  # Doit afficher la même valeur que spec.observation_specs[0].shape
        print(f"Agent state dim: {agent.state_dim}")  # Doit correspondre

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
                # state = obs
                state = obs.flatten()  # shape (8000,)
            else:
                state = np.zeros(state_dim, dtype=np.float32)
                done = True

            while not done and steps_count < max_steps:
                steps_count += 1

                # Action DDPG
                action_cont = agent.act(state, add_noise=True)

                print("action_cont shape=", action_cont.shape)

                # On ignore la discrète => fixons disc=0
                action_tuple = ActionTuple()
                action_tuple.add_continuous(np.array(action_cont, dtype=np.float32))  # shape (1,3)
                action_tuple.add_discrete(np.array([[0]], dtype=np.int32))  # fix disc=0

                env.set_actions(behavior_name, action_tuple)
                env.step()

                decision_steps, terminal_steps = env.get_steps(behavior_name)
                print(f"decision_steps 0 ={decision_steps.obs[0].shape}, terminal_steps={len(terminal_steps)}")

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
                print(f"State shape after flattening: {state.shape}")  # Devrait être (8000,)

                # Entraîner l'agent
                agent.step(state, action_cont, reward, next_state, float(done))

                state = next_state

            print(f"[Episode {episode}] Reward={ep_reward:.2f}")

            # On loggue la reward épisode dans TensorBoard
            agent.writer.add_scalar("Episode/Reward", ep_reward, episode)

    finally:

        env.close()
        print("DDPG Training done.")


if __name__ == "__main__":
    main()
