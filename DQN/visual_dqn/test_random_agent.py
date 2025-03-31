import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
import matplotlib.pyplot as plt
from mlagents_envs.base_env import ActionTuple


def test_random_agent(env_path, num_episodes=100, max_steps=1000):
    # Configuration du canal pour le time scale
    channel = EngineConfigurationChannel()
    env = UnityEnvironment(file_name=env_path, side_channels=[channel])
    channel.set_configuration_parameters(time_scale=1.0)  # Time scale normal pour bien voir

    env.reset()
    behavior_name = list(env.behavior_specs.keys())[0]
    spec = env.behavior_specs[behavior_name]

    # Récupération des dimensions des actions
    action_spec = spec.action_spec

    print(f"Action spec:")
    print(f"- Continuous size: {action_spec.continuous_size}")
    print(f"- Discrete branches: {action_spec.discrete_branches}")

    scores = []

    for episode in range(num_episodes):
        env.reset()
        episode_score = 0

        for step in range(max_steps):
            # Obtention des observations
            decision_steps, terminal_steps = env.get_steps(behavior_name)

            # Nombre d'agents actifs
            num_agents = len(decision_steps)

            if num_agents > 0:
                # Génération d'actions aléatoires
                continuous_actions = np.random.uniform(-1, 1, (num_agents, action_spec.continuous_size))
                if len(action_spec.discrete_branches) > 0:
                    discrete_actions = np.random.randint(0, action_spec.discrete_branches[0],
                                                         (num_agents, len(action_spec.discrete_branches)))
                else:
                    discrete_actions = None

                # Création de l'ActionTuple
                action = ActionTuple(continuous=continuous_actions, discrete=discrete_actions)

                # Exécution de l'action
                env.set_actions(behavior_name, action)
                env.step()

                # Calcul de la récompense
                if len(terminal_steps) > 0:
                    reward = terminal_steps.reward[0]
                    episode_score += reward
                    break
                else:
                    reward = decision_steps.reward[0]
                    episode_score += reward

        scores.append(episode_score)
        print(f"Episode {episode + 1}/{num_episodes}, Score: {episode_score:.2f}")

    env.close()
    return scores


def _plot_random_results(scores):
    plt.figure(figsize=(10, 6))
    plt.plot(scores, label='Scores')
    plt.axhline(y=np.mean(scores), color='r', linestyle='--', label=f'Moyenne: {np.mean(scores):.2f}')
    plt.title('Scores avec Agent Aléatoire')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.savefig('random_agent_scores.png')
    plt.close()


if __name__ == "__main__":
    env_path = "../FoodCollectorVisual/FoodCollector.exe"  # Ajustez le chemin selon votre configuration
    scores = test_random_agent(env_path)
    # plot_random_results(scores)

    print("\nStatistiques de l'agent aléatoire:")
    print(f"Score moyen: {np.mean(scores):.2f}")
    print(f"Score maximum: {np.max(scores):.2f}")
    print(f"Score minimum: {np.min(scores):.2f}")
    print(f"Écart-type: {np.std(scores):.2f}")