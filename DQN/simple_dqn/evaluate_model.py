import numpy as np
import torch
import time
import os
import warnings
import logging
from dqn_agent import QNetwork, DQNAgent

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.base_env import ActionTuple

# Configurer les avertissements pour ignorer les messages du canal inconnu
warnings.filterwarnings("ignore", message="Unknown side channel data received")
# Alternative: rediriger les avertissements ML-Agents vers un niveau inférieur
logging.getLogger("mlagents").setLevel(logging.ERROR)

def evaluate_model(env_path, model_path, num_episodes=10, max_steps=1000, render=True):
    """
    Évalue un modèle DQN déjà entraîné dans l'environnement FoodCollector.
    
    Params
    ======
        env_path (str): Chemin vers l'exécutable Unity
        model_path (str): Chemin vers le fichier de modèle entraîné (.pth)
        num_episodes (int): Nombre d'épisodes d'évaluation
        max_steps (int): Nombre maximal d'étapes par épisode
        render (bool): Si True, exécute en temps réel pour visualisation
    
    Returns
    ======
        scores (list): Liste des scores totaux par épisode
    """
    # Configurer le canal pour la configuration du moteur Unity
    channel = EngineConfigurationChannel()
    
    # Initialiser l'environnement Unity
    print(f"Démarrage de l'environnement Unity: {env_path}")
    env = UnityEnvironment(file_name=env_path, side_channels=[channel], no_graphics=not render)
    
    # Si render est True, définir la timescale à 1.0 pour visualisation en temps réel
    # Sinon, accélérer la simulation
    if render:
        channel.set_configuration_parameters(time_scale=1.0, width=1280, height=720)
    else:
        channel.set_configuration_parameters(time_scale=2.0)
    
    # Réinitialiser l'environnement
    env.reset()
    
    # Obtenir le behavior name par défaut
    behavior_name = list(env.behavior_specs)[0]
    print(f"Behavior name: {behavior_name}")
    
    # Obtenir les spécifications du comportement
    spec = env.behavior_specs[behavior_name]
    
    # Déterminer si les actions sont continues ou discrètes
    if spec.action_spec.continuous_size > 0:
        print(f"Actions continues: {spec.action_spec.continuous_size}")
        print(f"Actions discrètes: {spec.action_spec.discrete_branches}")
        
        # Même logique de discrétisation que dans train_food_collector.py
        discrete_actions = 3 * 3 * 3 * 2  # 3^3 * 2 = 54 actions discrètes possibles
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
    
    # Charger le modèle entraîné
    print(f"Chargement du modèle: {model_path}")
    if os.path.exists(model_path):
        agent.qnetwork_local.load_state_dict(torch.load(model_path))
        print("Modèle chargé avec succès.")
    else:
        print(f"Erreur: Le fichier de modèle {model_path} n'existe pas.")
        env.close()
        return []
    
    # Liste pour stocker les scores
    scores = []
    
    # Début de l'évaluation
    print("Début de l'évaluation...")
    for i_episode in range(1, num_episodes+1):
        env.reset()
        
        # Obtenir l'état initial
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        agent_ids = decision_steps.agent_id
        num_agents = len(agent_ids)
        
        if num_agents == 0:
            print("Aucun agent disponible. Vérifiez l'environnement Unity.")
            break
        
        print(f"Nombre d'agents: {num_agents}")
        
        # Pour simplifier, nous évaluons avec le premier agent mais envoyons des actions à tous
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
            # Sélectionner une action (sans exploration)
            action_idx = agent.act(state, eps=0.0)
            
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
            
            # Si render est True, ajouter un petit délai pour la visualisation
            if render and t % 5 == 0:
                time.sleep(0.01)
            
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
            
            # Mettre à jour l'état et le score
            state = next_state
            score += reward
            
            if done:
                break
        
        # Sauvegarder le score
        scores.append(score)
        
        # Afficher les statistiques
        print(f'Épisode {i_episode}/{num_episodes}\tScore: {score:.2f}')
    
    # Calculer et afficher le score moyen
    mean_score = np.mean(scores)
    print(f'\nScore moyen sur {num_episodes} épisodes: {mean_score:.2f}')
    
    # Fermer l'environnement
    env.close()
    
    return scores

if __name__ == "__main__":
    # Chemin vers l'exécutable Unity
    #env_path = "../FoodCollector_Windows_x86_64/UnityEnvironment.exe"
    env_path = "../FoodCollector1Agent_Win64/FoodCollector.exe"
    
    # Chemin vers le modèle entraîné
    model_path = "checkpoints/model_final_1000epds.pth"

    
    # Si le modèle final n'existe pas, rechercher le dernier point de contrôle
    if not os.path.exists(model_path):
        checkpoint_files = [f for f in os.listdir("checkpoints") if f.startswith("checkpoint_episode_")]
        if checkpoint_files:
            # Trier les fichiers par numéro d'épisode
            checkpoint_files.sort(key=lambda f: int(f.split("_")[2].split(".")[0]))
            model_path = os.path.join("checkpoints", checkpoint_files[-1])
            print(f"Modèle final non trouvé, utilisation du dernier point de contrôle: {model_path}")
        else:
            print("Aucun modèle trouvé. Veuillez d'abord entraîner un agent.")
            exit(1)
    
    # Évaluer le modèle
    evaluate_model(env_path, model_path, num_episodes=5, render=True)