import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import os
import warnings
import logging
import uuid
from collections import deque
from gru_dqn_agent import GRUDQNAgent, GRUQNetwork, device, LR
from dqn_agent import DQNAgent, QNetwork
import torch.optim as optim

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel import SideChannel

# Configurer les avertissements pour ignorer les messages du canal inconnu
warnings.filterwarnings("ignore")
# Alternative: rediriger les avertissements ML-Agents vers un niveau inférieur
logging.getLogger("mlagents").setLevel(logging.ERROR)

# Définir des hyperparamètres d'entraînement pour le fine-tuning
ADDITIONAL_EPISODES = 500  # Nombre d'épisodes supplémentaires
MAX_STEPS = 1000          # Nombre maximal d'étapes par épisode
EPS_START = 0.1           # Valeur initiale d'epsilon pour l'exploration (réduite)
EPS_END = 0.01            # Valeur minimale d'epsilon
EPS_DECAY = 0.99          # Facteur de décroissance d'epsilon
TARGET_SCORE = 30         # Score moyen à atteindre pour considérer l'environnement résolu
TIME_SCALE = 2.0         # Vitesse de l'environnement

class CustomChannel(SideChannel):
    def __init__(self):
        super().__init__(uuid.UUID("a1d8f7b7-cec8-50f9-b78b-d3e165a78520"))

    def on_message_received(self, msg: bytes) -> None:
        pass

def finetune_model(env_path, model_path, network_type='dqn', additional_episodes=ADDITIONAL_EPISODES, 
                  max_steps=MAX_STEPS, eps_start=EPS_START, eps_end=EPS_END, eps_decay=EPS_DECAY, 
                  target_score=TARGET_SCORE, time_scale=TIME_SCALE):
    """
    Continue l'entraînement d'un modèle DQN ou GRU-DQN déjà entraîné.
    
    Params
    ======
        env_path (str): Chemin vers l'exécutable Unity
        model_path (str): Chemin vers le fichier de modèle entraîné (.pth)
        network_type (str): Type de réseau à utiliser ('dqn' ou 'gru')
        additional_episodes (int): Nombre d'épisodes supplémentaires d'entraînement
        max_steps (int): Nombre maximal d'étapes par épisode
        eps_start (float): Valeur initiale d'epsilon pour l'exploration
        eps_end (float): Valeur minimale d'epsilon
        eps_decay (float): Facteur de décroissance d'epsilon
        target_score (float): Score moyen à atteindre pour considérer l'entraînement réussi
    
    Returns
    ======
        scores (list): Liste des scores totaux par épisode
        losses (list): Liste des pertes moyennes par épisode
    """
    # Vérifier si le modèle existe
    if not os.path.exists(model_path):
        print(f"Erreur: Le fichier de modèle {model_path} n'existe pas.")
        return [], []
    
    # Configurer le canal pour la configuration du moteur Unity
    channel = EngineConfigurationChannel()
    
    # Initialiser l'environnement Unity
    print(f"Démarrage de l'environnement Unity: {env_path}")
    env = UnityEnvironment(file_name=env_path, side_channels=[channel, CustomChannel()], no_graphics=True)
    
    # Définir la timescale pour accélérer l'entraînement
    channel.set_configuration_parameters(time_scale=time_scale)
    
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
        
        # Même logique de discrétisation que dans train_food_collector.py
        discrete_actions = 3 * 3 * 3  # 3^3 = 27 actions discrètes possibles
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
    
    # Initialiser l'agent selon le type de réseau
    if network_type.lower() == 'gru':
        print("Utilisation de l'agent GRU-DQN")
        agent = GRUDQNAgent(state_size=state_size, action_size=action_size, seed=42)
    else:
        print("Utilisation de l'agent DQN standard")
        agent = DQNAgent(state_size=state_size, action_size=action_size, seed=42)
    
    # Charger le modèle pré-entraîné
    print(f"Chargement du modèle: {model_path}")
    agent.qnetwork_local.load_state_dict(torch.load(model_path))
    agent.qnetwork_target.load_state_dict(torch.load(model_path))
    print("Modèle chargé avec succès, poursuite de l'entraînement.")
    
    # Liste pour stocker les scores
    scores = []
    # Déque pour calculer la moyenne sur les 100 derniers épisodes
    scores_window = deque(maxlen=100)
   
    losses = []
    
    eps = eps_start

    checkpoints_dir = 'checkpoints-finetune'
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    
    print("Début du fine-tuning...")
    for i_episode in range(1, additional_episodes+1):
        env.reset()
        
        # Obtenir l'état initial
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        agent_ids = decision_steps.agent_id
        num_agents = len(agent_ids)
        
        if num_agents == 0:
            print("Aucun agent disponible. Vérifiez l'environnement Unity.")
            break
        
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
        
        
        score = 0
    
        episode_losses = []
        
        
        for t in range(max_steps):
            # Sélectionner une action
            action_idx = agent.act(state, eps)
            
            # Convertir l'action discrète en action compatible avec l'environnement
            if spec.action_spec.continuous_size > 0:
                
                continuous_actions = np.zeros((num_agents, spec.action_spec.continuous_size))
                
                # On garde la possibilité d'avoir des actions discrètes au cas où l'environnement en a encore
                if len(spec.action_spec.discrete_branches) > 0:
                    discrete_actions_array = np.zeros((num_agents, len(spec.action_spec.discrete_branches)), dtype=np.int32)
                
                # Configuration de l'action pour l'agent principal
                move_forward = (action_idx // 9) - 1  # -1, 0, 1
                move_side = ((action_idx // 3) % 3) - 1  # -1, 0, 1
                rotate = (action_idx % 3) - 1  # -1, 0, 1
                
                # Appliquer la même action à tous les agents pour simplifier
                for i in range(num_agents):
                    continuous_actions[i, 0] = move_forward
                    continuous_actions[i, 1] = move_side
                    continuous_actions[i, 2] = rotate
                
                # Créer les actions en utilisant la méthode correcte
                action_tuple = ActionTuple()
                action_tuple.add_continuous(continuous_actions)
                
                # Ajouter les actions discrètes seulement si l'environnement en a
                if len(spec.action_spec.discrete_branches) > 0:
                    action_tuple.add_discrete(discrete_actions_array)
                    
                env.set_actions(behavior_name, action_tuple)
            else:
                # Si uniquement des actions discrètes
                action_tuple = ActionTuple()
                action_tuple.add_discrete(np.array([action_idx], dtype=np.int32).reshape(1, 1))
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
            
            # Récupérer la perte si disponible (après apprentissage)
            if hasattr(agent, 'last_loss') and agent.last_loss is not None:
                episode_losses.append(agent.last_loss)
            
            # Mettre à jour l'état et le score
            state = next_state
            score += reward
            
            if done:
                break
        
        # Mettre à jour epsilon (décroissance plus lente pour le fine-tuning)
        eps = max(eps_end, eps_decay*eps)
        
       
        scores_window.append(score)
        scores.append(score)
        
        
        if episode_losses:
            losses.append(np.mean(episode_losses))
        else:
            losses.append(0.0)  # Pas de perte si pas d'apprentissage
        
       
        print(f'\rÉpisode {i_episode}\tScore: {score:.2f}\tScore moyen: {np.mean(scores_window):.2f}', end="")
        
        
        if i_episode % 100 == 0:
            print(f'\rÉpisode {i_episode}\tScore moyen: {np.mean(scores_window):.2f}')
            # Sauvegarder le modèle
            torch.save(agent.qnetwork_local.state_dict(), f'{checkpoints_dir}/finetuned_{network_type}_{timestamp}_ep{i_episode}.pth')
        
        # Vérifier si l'environnement est résolu
        if np.mean(scores_window) >= target_score:
            print(f'\nEnvironnement résolu en {i_episode} épisodes!\tScore moyen: {np.mean(scores_window):.2f}')
            # Sauvegarder le modèle final
            torch.save(agent.qnetwork_local.state_dict(), f'{checkpoints_dir}/finetuned_{network_type}_{timestamp}_final.pth')
            break
    
    
    env.close()
    
    # Sauvegarder le modèle final si non sauvegardé
    if additional_episodes > 0:
        torch.save(agent.qnetwork_local.state_dict(), f'{checkpoints_dir}/finetuned_{network_type}_{timestamp}_final.pth')
    
    return scores, losses

def plot_scores(scores, losses=None, title_prefix="Fine-tuning"):
    """Affiche un graphique des scores et optionnellement des pertes pendant l'entraînement."""
    
    if not os.path.exists('figures'):
        os.makedirs('figures')
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
   
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Épisode')
    plt.title(f'{title_prefix} - Scores')
    
    
    window_size = min(100, len(scores))
    if window_size > 0:
        moving_avg = np.convolve(scores, np.ones((window_size,))/window_size, mode='valid')
        plt.plot(np.arange(len(moving_avg)), moving_avg, color='red')
    
    plt.savefig(f'figures/finetuning_scores_{timestamp}.png')
    plt.show()
    
    
    if losses is not None and len(losses) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(len(losses)), losses)
        plt.ylabel('Loss')
        plt.xlabel('Épisode')
        plt.title(f'{title_prefix} - Pertes')
        
        # Calculer la moyenne mobile sur 100 épisodes (ou moins si moins de données)
        window_size = min(100, len(losses))
        if window_size > 0:
            moving_avg_loss = np.convolve(losses, np.ones((window_size,))/window_size, mode='valid')
            plt.plot(np.arange(len(moving_avg_loss)), moving_avg_loss, color='red')
        
        plt.savefig(f'figures/finetuning_loss_{timestamp}.png')
        plt.show()

if __name__ == "__main__":
    # Chemin vers l'exécutable Unity
    env_path = "../FoodCollectorOneAgent/FoodCollector.exe"
    
    # Chemin vers le modèle pré-entraîné
    model_path = "checkpoints/model_final-15m.pth"
    
    # Type de réseau utilisé lors de l'entraînement initial
    network_type = 'dqn'  # 'dqn' ou 'gru', selon le modèle que vous souhaitez fine-tuner
    
    start_time = time.time()
    
    # Continuer l'entraînement du modèle
    scores, losses = finetune_model(
        env_path=env_path,
        model_path=model_path,
        network_type=network_type,
        additional_episodes=500,  # Nombre d'épisodes supplémentaires
        eps_start=0.1,            # Commencer avec moins d'exploration
        target_score=30,           # Objectif de score plus élevé
        time_scale=5.0
    )
    
    # Afficher les résultats
    plot_scores(scores, losses)
    
    end_time = time.time()
    execution_time = end_time - start_time

    if execution_time > 3600:
        hours = execution_time // 3600
        minutes = (execution_time % 3600) // 60
        seconds = execution_time % 60
        print(f"Temps d'exécution: {hours:.0f}h{minutes:.0f}minutes {seconds:.0f} secondes")
    elif execution_time > 60:
        minutes = execution_time // 60
        seconds = execution_time % 60
        print(f"Temps d'exécution: {minutes:.0f} minutes {seconds:.0f} secondes")
    else:
        print(f"Temps d'exécution: {execution_time:.2f} secondes")
