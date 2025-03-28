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
        pass

def main():
    # 1) Lancement de l'env
    config_channel = EngineConfigurationChannel()
    env = UnityEnvironment(
        file_name="environment/GridFoodCollector.exe",
        side_channels=[config_channel, CustomChannel()],
        worker_id=1,
        seed=0,
        no_graphics=True
    )
    # Définir la timescale pour accélérer l'entraînement
    try :
        config_channel.set_configuration_parameters(time_scale=20.0)
        env.reset()

        behavior_name = list(env.behavior_specs)[0]
        spec = env.behavior_specs[behavior_name]

        assert spec.action_spec.continuous_size == 3, "On attend 3 actions continues"
        # On ignore la discrète => fix=0

        # 2) Créer l'agent DDPG CNN
        agent = DDPGAgent(in_channels=5, action_dim=3, device="cuda" if torch.cuda.is_available() else "cpu")

        max_episodes = 100
        max_steps = 100

        # Pour logger la reward dans TensorBoard, on utilise l'agent.writer
        from torch.utils.tensorboard import SummaryWriter
        writer = agent.writer  # on réutilise le SummaryWriter existant

        for episode in range(max_episodes):
            env.reset()
            decision_steps, terminal_steps = env.get_steps(behavior_name)

            done = False
            episode_reward = 0.0
            steps_count = 0

            # Récup obs init
            if len(decision_steps) > 0:
                obs_img = decision_steps.obs[0][0]  # => shape(5,40,40)
            else:
                obs_img = np.zeros((5, 40, 40), dtype=np.float32)
                done = True

            # while not done and steps_count < max_steps:
            while not done :
                steps_count += 1

                # Sélection action
                action_cont = agent.act(obs_img, add_noise=True)  # => shape(3,)

                # Construit l'ActionTuple
                at = ActionTuple()
                at.add_continuous(np.array([action_cont], dtype=np.float32))  # => (1,3)
                at.add_discrete(np.array([[0]], dtype=np.int32))  # fix=0

                env.set_actions(behavior_name, at)
                env.step()

                decision_steps, terminal_steps = env.get_steps(behavior_name)

                # reward = 0.0
                next_obs_img = obs_img
                done = False

                if len(decision_steps) > 0:
                    obs_next = decision_steps.obs[0][0]
                    next_obs_img = obs_next
                    reward = decision_steps.reward[0]

                if len(terminal_steps) > 0:
                    for agent_id in terminal_steps:
                        reward = terminal_steps[agent_id].reward
                    done = True

                # print(f"step reward={reward:.2f}, done={done}")

                episode_reward += reward

                # Stocker la transition + learn
                agent.step(obs_img, action_cont, reward, next_obs_img, float(done))

                obs_img = next_obs_img

            # Fin d'épisode
            print(f"[Episode {episode}] reward={episode_reward:.2f}, steps={steps_count}")
            writer.add_scalar("EpisodeReward", episode_reward, episode)
    finally:

        env.close()
        print("Entraînement DDPG CNN terminé.")

if __name__ == "__main__":
    main()
