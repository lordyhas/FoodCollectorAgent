import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import os
import uuid
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel import SideChannel
import argparse


class CustomChannel(SideChannel):
    def __init__(self):
        super().__init__(uuid.UUID("a1d8f7b7-cec8-50f9-b78b-d3e165a78520"))

    def on_message_received(self, msg: bytes) -> None:
        pass


def analyze_environment(env_path, episodes=5, steps_per_episode=100, graphics=False, time_scale=1.0):
    """
    Analyse l'environnement FoodCollector et affiche un résumé des informations importantes.
    
    Params:
    ======
        env_path (str): Chemin vers l'exécutable Unity
        episodes (int): Nombre d'épisodes à exécuter pour l'analyse
        steps_per_episode (int): Nombre d'étapes par épisode
        graphics (bool): Afficher les graphiques ou non
        time_scale (float): Vitesse de simulation
    """
    print("\n" + "="*50)
    print("ANALYSE DE L'ENVIRONNEMENT FOOD COLLECTOR")
    print("="*50 + "\n")
    
    # Configurer le canal pour la configuration du moteur Unity
    channel = EngineConfigurationChannel()
    
    # Initialiser l'environnement Unity
    print(f"Connexion à l'environnement Unity: {env_path}")
    env = UnityEnvironment(file_name=env_path, side_channels=[channel, CustomChannel()], no_graphics=not graphics)
    
    # Définir la timescale
    channel.set_configuration_parameters(time_scale=time_scale)
    
    # Réinitialiser l'environnement
    env.reset()
    
    # Obtenir le behavior name par défaut
    behavior_name = list(env.behavior_specs)[0]
    print(f"Behavior name identifié: {behavior_name}")
    
    # Obtenir les spécifications du comportement
    spec = env.behavior_specs[behavior_name]
    
    # Analyser les spécifications d'action
    print("\n" + "-"*50)
    print("SPÉCIFICATIONS DES ACTIONS")
    print("-"*50)
    
    # Analyser les actions continues
    if spec.action_spec.continuous_size > 0:
        print(f"✓ Actions continues: {spec.action_spec.continuous_size} dimensions")
        print("  └─ Généralement dans FoodCollector:")
        print("     ├─ Dimension 0: Mouvement avant/arrière")
        print("     ├─ Dimension 1: Mouvement latéral")
        print("     └─ Dimension 2: Rotation")
    else:
        print("✗ Pas d'actions continues")
    
    # Analyser les actions discrètes
    if len(spec.action_spec.discrete_branches) > 0:
        print(f"✓ Actions discrètes: {len(spec.action_spec.discrete_branches)} branches")
        for i, branch_size in enumerate(spec.action_spec.discrete_branches):
            print(f"  ├─ Branche {i}: {branch_size} actions possibles")
            if i == 0 and branch_size == 2:
                print("     └─ Probablement contrôle du laser (0: inactif, 1: actif)")
    else:
        print("✗ Pas d'actions discrètes")
    
    # Calculer l'espace d'action total
    if spec.action_spec.continuous_size > 0:
        # Pour les actions continues, nous considérons une discrétisation 3x3x3
        discrete_actions = 3 ** spec.action_spec.continuous_size
        print(f"\n✓ Espace d'action effectif (après discrétisation): {discrete_actions} actions")
        if len(spec.action_spec.discrete_branches) > 0:
            discrete_actions *= np.prod(spec.action_spec.discrete_branches)
            print(f"  └─ Avec actions discrètes: {discrete_actions} actions combinées")
    else:
        if len(spec.action_spec.discrete_branches) > 0:
            discrete_actions = np.prod(spec.action_spec.discrete_branches)
            print(f"\n✓ Espace d'action total: {discrete_actions} actions")
        else:
            print("✗ Aucune action disponible!")
    
    # Analyser les observations
    print("\n" + "-"*50)
    print("SPÉCIFICATIONS DES OBSERVATIONS")
    print("-"*50)
    
    shapes = []
    obs_types = []
    state_size = 0
    
    # Pour chaque type d'observation
    for i, obs_spec in enumerate(spec.observation_specs):
        shape = obs_spec.shape
        shapes.append(shape)
        
        if len(shape) == 1:
            # Observation vectorielle
            state_size += shape[0]
            obs_types.append("vecteur")
            print(f"✓ Observation {i}: Vecteur de taille {shape[0]}")
            
            # Indication de contenu typique dans FoodCollector
            if shape[0] == 172:
                print("  └─ Contenu typique pour un vecteur de taille 172:")
                print("     ├─ Position de l'agent")
                print("     ├─ Raycast autour de l'agent (détection d'objets)")
                print("     ├─ Vélocité")
                print("     └─ Collectibles visibles")
        elif len(shape) == 3:
            # Observation visuelle (image)
            obs_types.append("image")
            flat_size = shape[0] * shape[1] * shape[2]
            state_size += flat_size
            print(f"✓ Observation {i}: Image de taille {shape[0]}x{shape[1]}x{shape[2]}")
            print(f"  └─ Dimensions aplaties: {flat_size}")
        else:
            obs_types.append("inconnu")
            print(f"? Observation {i}: Type inconnu de forme {shape}")
    
    print(f"\n✓ Taille totale de l'état (aplati): {state_size}")
    
    # Analyser les récompenses et pénalités en exécutant quelques épisodes
    print("\n" + "-"*50)
    print("ANALYSE DES RÉCOMPENSES")
    print("-"*50)
    
    # Stocker les informations sur les récompenses
    rewards = []
    episode_lengths = []
    
    # Exécuter quelques épisodes pour analyser
    for episode in range(episodes):
        env.reset()
        episode_reward = 0
        episode_length = 0
        
        # Obtenir l'état initial
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        agent_ids = decision_steps.agent_id
        
        if len(agent_ids) == 0:
            print("Aucun agent disponible. Vérifiez l'environnement Unity.")
            break
            
        agent_id = agent_ids[0]
        
        for step in range(steps_per_episode):
            # Prendre une action aléatoire
            if spec.action_spec.continuous_size > 0:
                # Actions continues aléatoires entre -1 et 1
                continuous_actions = np.random.uniform(-1, 1, (len(agent_ids), spec.action_spec.continuous_size))
                
                # Préparer les actions
                action_tuple = ActionTuple()
                action_tuple.add_continuous(continuous_actions)
                
                # Ajouter des actions discrètes si nécessaire
                if len(spec.action_spec.discrete_branches) > 0:
                    discrete_actions_array = np.zeros((len(agent_ids), len(spec.action_spec.discrete_branches)), dtype=np.int32)
                    for i, branch_size in enumerate(spec.action_spec.discrete_branches):
                        discrete_actions_array[:, i] = np.random.randint(0, branch_size, len(agent_ids))
                    action_tuple.add_discrete(discrete_actions_array)
            else:
                # Uniquement des actions discrètes
                action_tuple = ActionTuple()
                discrete_actions_array = np.zeros((len(agent_ids), len(spec.action_spec.discrete_branches)), dtype=np.int32)
                for i, branch_size in enumerate(spec.action_spec.discrete_branches):
                    discrete_actions_array[:, i] = np.random.randint(0, branch_size, len(agent_ids))
                action_tuple.add_discrete(discrete_actions_array)
            
            # Exécuter l'action
            env.set_actions(behavior_name, action_tuple)
            env.step()
            
            # Obtenir le nouvel état et la récompense
            decision_steps, terminal_steps = env.get_steps(behavior_name)
            
            # Vérifier si l'agent a terminé
            done = False
            reward = 0
            
            if agent_id in terminal_steps:
                # L'agent a terminé
                reward = terminal_steps[agent_id].reward
                done = True
            elif agent_id in decision_steps:
                # L'agent continue
                reward = decision_steps[agent_id].reward
            else:
                # L'agent n'est pas disponible
                break
            
            episode_reward += reward
            episode_length += 1
            
            # Ajouter la récompense à la liste pour analyse
            rewards.append(reward)
            
            if done:
                break
        
        episode_lengths.append(episode_length)
        print(f"Épisode {episode+1}/{episodes}: Récompense totale = {episode_reward:.2f}, Étapes = {episode_length}")
    
    # Analyser les récompenses
    if rewards:
        min_reward = min(rewards)
        max_reward = max(rewards)
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        
        print(f"\n✓ Récompense minimale observée: {min_reward:.4f}")
        print(f"✓ Récompense maximale observée: {max_reward:.4f}")
        print(f"✓ Récompense moyenne: {mean_reward:.4f} ± {std_reward:.4f}")
        
        # Identification des événements de récompense
        if min_reward < 0:
            print("✓ Pénalités détectées - probablement pour:")
            print("  └─ Sortir de la zone de jeu")
            print("  └─ Collision avec des obstacles")
            print("  └─ Utilisation du laser sans nécessité")
        
        if max_reward > 0:
            print("✓ Récompenses positives détectées - probablement pour:")
            print("  └─ Collecte de nourriture")
            print("  └─ Utilisation efficace du laser contre des ennemis")
    
    # Résumons les informations clés pour l'entraînement
    print("\n" + "="*50)
    print("RÉSUMÉ DES INFORMATIONS CLÉS POUR L'ENTRAÎNEMENT")
    print("="*50)
    
    print(f"\n✓ Type d'environnement: FoodCollector (Unity ML-Agents)")
    print(f"✓ Nombre d'agents: {len(agent_ids)}")
    print(f"✓ Taille de l'espace d'état: {state_size}")
    print(f"✓ Espace d'action: {discrete_actions} actions possibles")
    
    # Estimer la complexité
    complexity = "Faible"
    if state_size > 100:
        if discrete_actions > 20:
            complexity = "Élevée"
        else:
            complexity = "Moyenne"
    
    print(f"✓ Complexité estimée: {complexity}")
    
    # Recommandations de configuration
    print("\n" + "-"*50)
    print("RECOMMANDATIONS DE CONFIGURATION")
    print("-"*50)
    
    if "image" in obs_types:
        print("! Observation de type image détectée - consider CNN ou traitement d'image")
    
    print(f"\n✓ Configuration recommandée pour DNN:")
    print(f"  ├─ Taille de l'entrée: {state_size}")
    print(f"  ├─ Couches cachées: [256, 128]")
    print(f"  └─ Sortie: {discrete_actions}")
    
    print(f"\n✓ Configuration recommandée pour GRU/LSTM:")
    print(f"  ├─ Taille de l'entrée: {state_size}")
    print(f"  ├─ Taille de l'état caché: 128 ou 256")
    print(f"  ├─ Couches: 1 ou 2")
    print(f"  └─ Sortie: {discrete_actions}")
    
    if complexity == "Élevée":
        print("\n✓ Hyperparamètres DQN recommandés pour complexité élevée:")
        print("  ├─ Taille du buffer de replay: 1e6")
        print("  ├─ Batch size: 64 ou 128")
        print("  ├─ Gamma (discount): 0.99")
        print("  ├─ Tau (soft update): 1e-3")
        print("  ├─ Learning rate: 5e-4")
        print("  ├─ Update every: 4")
        print("  └─ Sequence length (GRU/LSTM): 8 ou 16")
    else:
        print("\n✓ Hyperparamètres DQN recommandés pour complexité moyenne/faible:")
        print("  ├─ Taille du buffer de replay: 1e5")
        print("  ├─ Batch size: 64")
        print("  ├─ Gamma (discount): 0.99")
        print("  ├─ Tau (soft update): 1e-3")
        print("  ├─ Learning rate: 5e-4")
        print("  ├─ Update every: 4")
        print("  └─ Sequence length (GRU/LSTM): 4 ou 8")
    
    # Fermez l'environnement
    env.close()
    
    # Visualisations
    if rewards and graphics:
        plt.figure(figsize=(12, 6))
        
        # Distribution des récompenses
        plt.subplot(1, 2, 1)
        plt.hist(rewards, bins=20, alpha=0.7)
        plt.axvline(mean_reward, color='r', linestyle='--', alpha=0.5, label=f'Moyenne: {mean_reward:.4f}')
        plt.title('Distribution des récompenses')
        plt.xlabel('Récompense')
        plt.ylabel('Fréquence')
        plt.legend()
        
        # Évolution des récompenses par étape
        plt.subplot(1, 2, 2)
        cumsum = np.cumsum(rewards)
        steps = np.arange(1, len(rewards) + 1)
        plt.plot(steps, cumsum / steps, label='Récompense moyenne cumulée')
        plt.title('Évolution de la récompense moyenne')
        plt.xlabel('Étapes')
        plt.ylabel('Récompense moyenne')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    return {
        "state_size": state_size,
        "action_size": discrete_actions,
        "complexity": complexity,
        "obs_types": obs_types,
        "mean_reward": np.mean(rewards) if rewards else 0,
        "episode_length_avg": np.mean(episode_lengths) if episode_lengths else 0
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyser l\'environnement FoodCollector')
    parser.add_argument('--env_path', type=str, default="../FoodCollectorOneAgent/FoodCollector.exe",
                        help='Chemin vers l\'exécutable Unity FoodCollector')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Nombre d\'épisodes à exécuter pour l\'analyse')
    parser.add_argument('--steps', type=int, default=10,
                        help='Nombre d\'étapes maximum par épisode')
    parser.add_argument('--graphics', action='store_true',
                        help='Afficher les graphiques')
    parser.add_argument('--time_scale', type=float, default=2.0,
                        help='Vitesse de simulation Unity (1.0 = temps réel)')
    
    args = parser.parse_args()
    
    # Analyser l'environnement
    analyze_environment(
        env_path=args.env_path,
        episodes=args.episodes,
        steps_per_episode=args.steps,
        graphics=args.graphics,
        time_scale=args.time_scale
    ) 