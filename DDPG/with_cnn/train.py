import numpy as np
import torch
import uuid

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel import SideChannel
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel


from ddpg_agent import DDPGAgent

class CustomChannel(SideChannel):
    def __init__(self):
        super().__init__(uuid.UUID("a1d8f7b7-cec8-50f9-b78b-d3e165a78520"))
    def on_message_received(self, msg: bytes) -> None:
        pass

def main():
    # 1) Lancement de l'env Unity
    config_channel = EngineConfigurationChannel()
    try:
        env = UnityEnvironment(
            file_name="environment/GridFoodCollector.exe",
            side_channels=[config_channel, CustomChannel()],
            seed=0,
            no_graphics=True,
            worker_id=1
        )
        env.reset()

        behavior_name = list(env.behavior_specs)[0]
        spec = env.behavior_specs[behavior_name]

        # Vérifier 3 actions continues
        assert spec.action_spec.continuous_size == 3, "On attend 3 actions continues"
        # On ignore la discrète (ou fix=0)
        # Vérifier obs : c'est (1,5,40,40) => CNN

        # 2) Créer l'agent
        agent = DDPGAgent(in_channels=5, action_dim=3, device="cuda" if torch.cuda.is_available() else "cpu")

        max_episodes = 500
        max_steps = 1000

        for episode in range(max_episodes):
            env.reset()
            decision_steps, terminal_steps = env.get_steps(behavior_name)

            done = False
            episode_reward = 0.0
            steps_count = 0

            # Récup obs init (1,5,40,40)
            if len(decision_steps) > 0:
                obs_tensor = decision_steps.obs[0]  # shape (1,5,40,40)
                # On extract juste [0], => shape(5,40,40)
                state_img = obs_tensor[0]
            else:
                state_img = np.zeros((5, 40, 40), dtype=np.float32)
                done = True

            while not done and steps_count < max_steps:
                steps_count += 1

                # 3) Sélection d'action (3 actions) via l'agent CNN
                action_cont = agent.act(state_img, add_noise=True)  # shape(3,)

                # On ignore la partie discrète => fix=0
                at = ActionTuple()
                at.add_continuous(np.array([action_cont], dtype=np.float32))  # => shape (1,3)
                at.add_discrete(np.array([[0]], dtype=np.int32))  # => shape (1,1)

                # 4) Appliquer l'action
                env.set_actions(behavior_name, at)
                env.step()

                decision_steps, terminal_steps = env.get_steps(behavior_name)

                reward = 0.0
                next_state_img = state_img
                done = False

                if len(decision_steps) > 0:
                    # On a un nouveau obs => shape(1,5,40,40)
                    obs_next = decision_steps.obs[0][0]  # shape(5,40,40)
                    next_state_img = obs_next
                    reward = decision_steps.reward[0]

                if len(terminal_steps) > 0:
                    # L'agent est terminé
                    for agent_id in terminal_steps:
                        reward = terminal_steps[agent_id].reward
                    done = True

                episode_reward += reward
                # 5) Entraîner l'agent (stock la transition + learn)
                agent.step(state_img, action_cont, reward, next_state_img, float(done))

                state_img = next_state_img

            print(f"[Episode {episode}] reward={episode_reward:.2f}, steps={steps_count}")
    finally:

        env.close()
        print("Entraînement terminé.")

if __name__ == "__main__":
    main()
