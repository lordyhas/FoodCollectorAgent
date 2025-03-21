import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import os
import warnings
import logging
from collections import deque
from dqn_agent import DQNAgent

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.base_env import ActionTuple

# Configurer les avertissements pour ignorer les messages du canal inconnu
warnings.filterwarnings("ignore")
# Alternative: rediriger les avertissements ML-Agents vers un niveau inférieur
logging.getLogger("mlagents").setLevel(logging.ERROR)
# Be more specific with the logger name
logging.getLogger("mlagents_envs.environment").setLevel(logging.ERROR)

# Définir des hyperparamètres d'entraînement
NUM_EPISODES = 1000      # Nombre réduit pour les tests initiaux
MAX_STEPS = 1000         # Nombre maximal d'étapes par épisode
EPS_START = 1.0          # Valeur initiale d'epsilon
EPS_END = 0.01           # Valeur minimale d'epsilon
EPS_DECAY = 0.995        # Facteur de décroissance d'epsilon
TARGET_SCORE = 15        # Score moyen à atteindre pour considérer l'environnement résolu

def train_dqn(env_path, num_episodes=NUM_EPISODES, max_steps=MAX_STEPS, eps_start=EPS_START, 
              eps_end=EPS_END, eps_decay=EPS_DECAY, target_score=TARGET_SCORE):
    """
    Entraîne un agent DQN dans l'environnement FoodCollector de Unity.
    
    Params
    ======
        env_path (str): Chemin vers l'exécutable Unity
        num_episodes (int): Nombre maximal d'épisodes d'entraînement
        max_steps (int): Nombre maximal d'étapes par épisode
        eps_start (float): Valeur initiale d'epsilon pour l'exploration
        eps_end (float): Valeur minimale d'epsilon
        eps_decay (float): Facteur de décroissance d'epsilon
        target_score (float): Score moyen à atteindre pour considérer l'entraînement réussi
    
    Returns
    ======
        scores (list): Liste des scores totaux par épisode
    """
    # Configurer le canal pour la configuration du moteur Unity
    channel = EngineConfigurationChannel()
    
    # Initialiser l'environnement Unity
    print(f"Démarrage de l'environnement Unity: {env_path}")
    env = UnityEnvironment(file_name=env_path, side_channels=[channel], no_graphics=True)
    
    # Définir la timescale pour accélérer l'entraînement
    channel.set_configuration_parameters(time_scale=20.0)
    
    # Réinitialiser l'environnement
    env.reset()
    
    # Obtenir le behavior name par défaut
    behavior_name = list(env.behavior_specs)[0]
    print(f"Behavior name: {behavior_name}")
    
    # Obtenir les spécifications du comportement
    spec = env.behavior_specs[behavior_name]
    
    # Déterminer si les actions sont continues ou discrètes
    if spec.action_spec.continuous_size > 0:
        print(f"Ce script est conçu pour des actions discrètes, mais l'environnement utilise des actions continues.")
        print(f"Actions continues: {spec.action_spec.continuous_size}")
        print(f"Actions discrètes: {spec.action_spec.discrete_branches}")
        
        # Pour cet exemple, nous allons discrétiser les actions continues
        # FoodCollector a généralement 3 actions continues (mouvement avant, latéral, rotation)
        # et 1 action discrète (laser)
        
        # Nombre total d'actions discrètes (simplification pour l'exemple)
        # 3 niveaux pour chaque action continue (avant, arrière, rien / droite, gauche, rien / tourner droite, tourner gauche, rien)
        # + nombre d'actions discrètes existantes
        discrete_actions = 3 * 3 * 3 * 2  # 3^3 * 2 = 54 actions discrètes possibles
        print(f"Discrétisation des actions continues en {discrete_actions} actions discrètes")
    else:
        discrete_actions = sum(spec.action_spec.discrete_branches)
    
    # Déterminer la taille de l'espace d'observation
    state_size = 0
    for obs_spec in spec.observation_specs:
        if len(obs_spec.shape) == 1:
            # Observation vectorielle
            state_size += obs_spec.shape[0]
        elif len(obs_spec.shape) == 3:
            # Observation visuelle (image)
            state_size += obs_spec.shape[0] * obs_spec.shape[1] * obs_spec.shape[2]
        else:
            print(f"Observation de type non supporté détectée: {obs_spec.shape}")
    
    # Définition de action_size
    action_size = discrete_actions
        
    print(f"Taille de l'état: {state_size}, Taille de l'action: {action_size}")
    
    # Initialiser l'agent DQN
    agent = DQNAgent(state_size=state_size, action_size=action_size, seed=42)
    
    # Liste pour stocker les scores
    scores = []
    # Déque pour calculer la moyenne sur les 100 derniers épisodes
    scores_window = deque(maxlen=100)
    # Epsilon pour l'exploration
    eps = eps_start
    
    # Créer un dossier pour sauvegarder les modèles
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    
    # Début de l'entraînement
    print("Début de l'entraînement...")
    for i_episode in range(1, num_episodes+1):
        env.reset()
        
        # Obtenir l'état initial
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        agent_ids = decision_steps.agent_id
        num_agents = len(agent_ids)
        
        if num_agents == 0:
            print("Aucun agent disponible. Vérifiez l'environnement Unity.")
            break
        
        print(f" | Nombre d'agents: {num_agents}")
        
        # Pour simplifier, nous entraînons avec le premier agent mais envoyons des actions à tous
        agent_id = agent_ids[0]
        
        # Extraire et aplatir les observations
        state = []
        for obs in decision_steps[agent_id].obs:
            if len(obs.shape) == 1:
                state.extend(obs)
            elif len(obs.shape) == 3:
                # Aplatir l'image
                state.extend(obs.flatten())
            else:
                print(f"Observation ignorée de forme: {obs.shape}")
        
        state = np.array(state)
        
        # Initialiser le score pour cet épisode
        score = 0
        
        # Exécuter l'épisode
        for t in range(max_steps):
            # Sélectionner une action
            action_idx = agent.act(state, eps)
            
            # Convertir l'action discrète en action compatible avec l'environnement
            if spec.action_spec.continuous_size > 0:
                # Préparation des actions pour tous les agents
                continuous_actions = np.zeros((num_agents, spec.action_spec.continuous_size))
                discrete_actions_array = np.zeros((num_agents, len(spec.action_spec.discrete_branches)), dtype=np.int32)
                
                # Configuration de l'action pour l'agent principal
                move_forward = (action_idx // 18) - 1  # -1, 0, 1
                move_side = ((action_idx // 6) % 3) - 1  # -1, 0, 1
                rotate = ((action_idx // 2) % 3) - 1  # -1, 0, 1
                laser = action_idx % 2  # 0, 1
                
                # Appliquer la même action à tous les agents pour simplifier
                for i in range(num_agents):
                    continuous_actions[i, 0] = move_forward
                    continuous_actions[i, 1] = move_side
                    continuous_actions[i, 2] = rotate
                    if len(spec.action_spec.discrete_branches) > 0:
                        discrete_actions_array[i, 0] = laser
                
                # Créer les actions en utilisant la méthode correcte
                if len(spec.action_spec.discrete_branches) > 0:
                    # Pour les actions mixtes (continues et discrètes)
                    action_tuple = ActionTuple()
                    action_tuple.add_continuous(continuous_actions)
                    action_tuple.add_discrete(discrete_actions_array)
                    env.set_actions(behavior_name, action_tuple)
                else:
                    # Seulement des actions continues
                    action_tuple = ActionTuple()
                    action_tuple.add_continuous(continuous_actions)
                    env.set_actions(behavior_name, action_tuple)
            else:
                # Si uniquement des actions discrètes
                action_tuple = ActionTuple()
                action_tuple.add_discrete([action_idx])
                env.set_actions(behavior_name, action_tuple)
            
            # Exécuter l'action dans l'environnement
            env.step()
            
            # Obtenir le nouvel état, la récompense et si l'épisode est terminé
            decision_steps, terminal_steps = env.get_steps(behavior_name)
            
            # Vérifier si l'agent a terminé
            done = False
            if agent_id in terminal_steps:
                # L'agent a terminé
                reward = terminal_steps[agent_id].reward
                
                # Extraire et aplatir les observations
                next_state = []
                for obs in terminal_steps[agent_id].obs:
                    if len(obs.shape) == 1:
                        next_state.extend(obs)
                    elif len(obs.shape) == 3:
                        # Aplatir l'image
                        next_state.extend(obs.flatten())
                    else:
                        print(f"Observation ignorée de forme: {obs.shape}")
                
                next_state = np.array(next_state)
                done = True
            elif agent_id in decision_steps:
                # L'agent continue
                reward = decision_steps[agent_id].reward
                
                # Extraire et aplatir les observations
                next_state = []
                for obs in decision_steps[agent_id].obs:
                    if len(obs.shape) == 1:
                        next_state.extend(obs)
                    elif len(obs.shape) == 3:
                        # Aplatir l'image
                        next_state.extend(obs.flatten())
                    else:
                        print(f"Observation ignorée de forme: {obs.shape}")
                
                next_state = np.array(next_state)
            else:
                # L'agent n'est pas disponible (peut-être réinitialisé)
                break
            
            # Mettre à jour l'agent
            agent.step(state, action_idx, reward, next_state, done)
            
            # Mettre à jour l'état et le score
            state = next_state
            score += reward
            
            if done:
                break
        
        # Mettre à jour epsilon
        eps = max(eps_end, eps_decay*eps)
        
        # Sauvegarder le score
        scores_window.append(score)
        scores.append(score)
        
        # Afficher les statistiques
        print(f'\rÉpisode {i_episode}\tScore: {score:.2f}\tScore moyen: {np.mean(scores_window):.2f}', end="")
        
        # Afficher des statistiques détaillées tous les 100 épisodes
        if i_episode % 100 == 0:
            print(f'\rÉpisode {i_episode}\tScore moyen: {np.mean(scores_window):.2f}')
            # Sauvegarder le modèle
            torch.save(agent.qnetwork_local.state_dict(), f'checkpoints/checkpoint_episode_{i_episode}.pth')
        
        # Vérifier si l'environnement est résolu
        if np.mean(scores_window) >= target_score:
            print(f'\nEnvironnement résolu en {i_episode} épisodes!\tScore moyen: {np.mean(scores_window):.2f}')
            # Sauvegarder le modèle final
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoints/model_final-15score.pth')
            break
    
    # Fermer l'environnement
    env.close()
    
    return scores

def plot_scores(scores):
    """Affiche un graphique des scores pendant l'entraînement."""
    # Créer un dossier pour les figures si nécessaire
    if not os.path.exists('figures'):
        os.makedirs('figures')
    
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Épisode')
    plt.title('Scores pendant l\'entraînement DQN')
    
    # Calculer la moyenne mobile sur 100 épisodes
    moving_avg = np.convolve(scores, np.ones((100,))/100, mode='valid')
    plt.plot(np.arange(len(moving_avg)), moving_avg, color='red')
    
    plt.savefig('figures/scores_plot-15score.png')
    plt.show()

if __name__ == "__main__":
    # Chemin vers l'exécutable Unity
    #env_path = "../FoodCollector_Windows_x86_64/UnityEnvironment.exe"
    env_path = "../FoodCollector1Agent_Win64/FoodCollector.exe"
    start_time = time.time()
    # Entraîner l'agent
    scores = train_dqn(env_path, num_episodes=1000, max_steps=1000)
    end_time = time.time()
    # Afficher les résultats
    print(f"\nDurée totale de l'entraînement: {end_time - start_time:.2f} secondes")
    plot_scores(scores) 