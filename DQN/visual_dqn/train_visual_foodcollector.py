import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import os
import uuid
from collections import deque
from cnn_dqn_agent import CNNDQNAgent, device
import torch.optim as optim
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel import SideChannel
from datetime import datetime

# Hyperparamètres d'entraînement
NUM_EPISODES = 2000      # Nombre d'épisodes
MAX_STEPS = 1000         # Nombre maximal d'étapes par épisode
EPS_START = 1.0          # Valeur initiale d'epsilon
EPS_END = 0.01           # Valeur minimale d'epsilon
EPS_DECAY = 0.995        # Facteur de décroissance d'epsilon
TARGET_SCORE = 15        # Score moyen à atteindre pour considérer l'environnement résolu
TIME_SCALE = 5.0         # Vitesse de simulation (5x plus rapide que temps réel)
CHECKPOINT_EVERY = 100   # Sauvegarder le modèle tous les N épisodes


class CustomChannel(SideChannel):
    def __init__(self):
        super().__init__(uuid.UUID("a1d8f7b7-cec8-50f9-b78b-d3e165a78520"))

    def on_message_received(self, msg: bytes) -> None:
        pass


def train_cnn_dqn(env_path, num_episodes=NUM_EPISODES, max_steps=MAX_STEPS, eps_start=EPS_START, 
                 eps_end=EPS_END, eps_decay=EPS_DECAY, target_score=TARGET_SCORE, time_scale=TIME_SCALE,
                 checkpoint_every=CHECKPOINT_EVERY):
    """
    Entraîne un agent CNN-DQN dans l'environnement FoodCollector visuel de Unity.
    
    Params
    ======
        env_path (str): Chemin vers l'exécutable Unity
        num_episodes (int): Nombre maximal d'épisodes d'entraînement
        max_steps (int): Nombre maximal d'étapes par épisode
        eps_start (float): Valeur initiale d'epsilon pour l'exploration
        eps_end (float): Valeur minimale d'epsilon
        eps_decay (float): Facteur de décroissance d'epsilon
        target_score (float): Score moyen à atteindre pour considérer l'entraînement réussi
        time_scale (float): Vitesse de simulation Unity (1.0 = temps réel)
        checkpoint_every (int): Fréquence de sauvegarde du modèle
    
    Returns
    ======
        scores (list): Liste des scores totaux par épisode
    """
    # Configurer le canal pour la configuration du moteur Unity
    channel = EngineConfigurationChannel()
    
    # Initialiser l'environnement Unity
    print(f"Démarrage de l'environnement Unity visuel: {env_path}")
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
        print(f"Actions continues: {spec.action_spec.continuous_size}")
        print(f"Actions discrètes: {spec.action_spec.discrete_branches}")
        
        # Pour cet exemple, nous allons discrétiser les actions continues
        # FoodCollector a généralement 3 actions continues (mouvement avant, latéral, rotation)
        # et 1 action discrète (laser)
        
        # Nombre total d'actions discrètes (simplification pour l'exemple)
        # 3 niveaux pour chaque action continue (avant, arrière, rien / droite, gauche, rien / tourner droite, tourner gauche, rien)
        # + nombre d'actions discrètes existantes
        discrete_actions = 3 * 3 * 3  # 3^3 = 27 actions discrètes possibles sans laser
        if len(spec.action_spec.discrete_branches) > 0:
            discrete_actions *= spec.action_spec.discrete_branches[0]  # Multiplier par le nombre d'actions laser
        
        print(f"Discrétisation des actions continues en {discrete_actions} actions discrètes")
        action_size = discrete_actions
    else:
        discrete_actions = sum(spec.action_spec.discrete_branches)
        action_size = discrete_actions
    
    # Déterminer la forme de l'observation visuelle
    state_shape = None
    
    # Détecter automatiquement la forme des observations
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    if len(decision_steps) > 0:
        agent_id = decision_steps.agent_id[0]
        
        # Trouver les observations visuelles (de forme 3D)
        for i, obs in enumerate(decision_steps[agent_id].obs):
            if len(obs.shape) == 3:
                state_shape = obs.shape
                print(f"Observation visuelle détectée: forme {state_shape}")
                break
    
    if state_shape is None:
        raise ValueError("Aucune observation visuelle n'a été détectée. Assurez-vous que l'environnement fournit des observations visuelles.")
    
    print(f"Forme de l'état: {state_shape}, Taille de l'action: {action_size}")
    
    # Initialiser l'agent CNN-DQN
    agent = CNNDQNAgent(state_shape=state_shape, action_size=action_size, seed=42)
    print(f"Agent CNN-DQN initialisé avec des entrées de forme {state_shape}")
    
    # Liste pour stocker les scores
    scores = []
    # Déque pour calculer la moyenne sur les 100 derniers épisodes
    scores_window = deque(maxlen=100)
    # Liste pour stocker les pertes
    losses = []
    # Epsilon pour l'exploration
    eps = eps_start
    
    # Créer des dossiers pour sauvegarder les modèles et figures
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    
    if not os.path.exists('figures'):
        os.makedirs('figures')
    
    # Timestamp pour les noms de fichiers
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Début de l'entraînement
    print("Début de l'entraînement de l'agent CNN-DQN...")
    start_time = time.time()
    
    for i_episode in range(1, num_episodes+1):
        env.reset()
        
        # Obtenir l'état initial
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        agent_ids = decision_steps.agent_id
        
        if len(agent_ids) == 0:
            print("Aucun agent disponible. Vérifiez l'environnement Unity.")
            break
        
        # Pour simplifier, nous entraînons avec le premier agent mais envoyons des actions à tous
        agent_id = agent_ids[0]
        
        # Extraire l'observation visuelle
        for i, obs in enumerate(decision_steps[agent_id].obs):
            if len(obs.shape) == 3:
                state = obs
                break
        
        # Initialiser le score pour cet épisode
        score = 0
        # Initialiser la perte pour cet épisode
        episode_losses = []
        
        # Exécuter l'épisode
        for t in range(max_steps):
            # Sélectionner une action
            action_idx = agent.act(state, eps)
            
            # Convertir l'action discrète en action compatible avec l'environnement
            if spec.action_spec.continuous_size > 0:
                # Préparation des actions pour tous les agents
                continuous_actions = np.zeros((len(agent_ids), spec.action_spec.continuous_size))
                
                # On garde la possibilité d'avoir des actions discrètes au cas où l'environnement en a encore
                if len(spec.action_spec.discrete_branches) > 0:
                    discrete_actions_array = np.zeros((len(agent_ids), len(spec.action_spec.discrete_branches)), dtype=np.int32)
                
                # Configuration de l'action pour l'agent principal
                move_forward = (action_idx // 9) - 1  # -1, 0, 1
                move_side = ((action_idx // 3) % 3) - 1  # -1, 0, 1
                rotate = (action_idx % 3) - 1  # -1, 0, 1
                
                # Appliquer la même action à tous les agents pour simplifier
                for i in range(len(agent_ids)):
                    continuous_actions[i, 0] = move_forward
                    continuous_actions[i, 1] = move_side
                    continuous_actions[i, 2] = rotate
                
                # Créer les actions en utilisant la méthode correcte
                action_tuple = ActionTuple()
                action_tuple.add_continuous(continuous_actions)
                
                # Ajouter les actions discrètes seulement si l'environnement en a
                if len(spec.action_spec.discrete_branches) > 0:
                    # Si l'action est assez grande pour inclure l'action discrète (laser)
                    if action_idx >= 27:
                        use_laser = 1  # Activer le laser
                    else:
                        use_laser = 0  # Ne pas utiliser le laser
                    
                    # Appliquer l'action discrète à tous les agents
                    for i in range(len(agent_ids)):
                        discrete_actions_array[i, 0] = use_laser
                    
                    # Ajouter les actions discrètes au tuple d'action
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
            reward = 0
            next_state = None
            
            if agent_id in terminal_steps:
                # L'agent a terminé
                reward = terminal_steps[agent_id].reward
                
                # Extraire l'observation visuelle
                for i, obs in enumerate(terminal_steps[agent_id].obs):
                    if len(obs.shape) == 3:
                        next_state = obs
                        break
                
                done = True
            elif agent_id in decision_steps:
                # L'agent continue
                reward = decision_steps[agent_id].reward
                
                # Extraire l'observation visuelle
                for i, obs in enumerate(decision_steps[agent_id].obs):
                    if len(obs.shape) == 3:
                        next_state = obs
                        break
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
        
        # Mettre à jour epsilon
        eps = max(eps_end, eps_decay*eps)
        
        # Sauvegarder le score
        scores_window.append(score)
        scores.append(score)
        
        # Sauvegarder la perte moyenne de l'épisode
        if episode_losses:
            losses.append(np.mean(episode_losses))
        else:
            losses.append(0.0)  # Pas de perte si pas d'apprentissage
        
        # Afficher les statistiques
        print(f'\rÉpisode {i_episode}\tScore: {score:.2f}\tScore moyen: {np.mean(scores_window):.2f}\tEpsilon: {eps:.3f}', end="")
        
        # Afficher des statistiques détaillées tous les checkpoint_every épisodes
        if i_episode % checkpoint_every == 0:
            print(f'\rÉpisode {i_episode}\tScore moyen: {np.mean(scores_window):.2f}\tTemps écoulé: {(time.time() - start_time)/60:.1f} min')
            
            # Sauvegarder le modèle
            checkpoint_path = f'checkpoints/cnn_dqn_checkpoint_episode_{i_episode}_{timestamp}.pth'
            torch.save(agent.qnetwork_local.state_dict(), checkpoint_path)
            print(f"Modèle sauvegardé: {checkpoint_path}")
            
            # Tracer et sauvegarder les graphiques
            plot_scores(scores, losses, f'figures/cnn_dqn_training_episode_{i_episode}_{timestamp}.png')
        
        # Vérifier si l'environnement est résolu
        if np.mean(scores_window) >= target_score:
            print(f'\nEnvironnement résolu en {i_episode} épisodes!\tScore moyen: {np.mean(scores_window):.2f}')
            
            # Sauvegarder le modèle final
            final_path = f'checkpoints/cnn_dqn_model_final_{timestamp}.pth'
            torch.save(agent.qnetwork_local.state_dict(), final_path)
            print(f"Modèle final sauvegardé: {final_path}")
            
            # Tracer et sauvegarder les graphiques finaux
            plot_scores(scores, losses, f'figures/cnn_dqn_training_final_{timestamp}.png')
            break
    
    # Fermer l'environnement
    env.close()
    
    # Si l'environnement n'a pas été résolu, sauvegarder quand même le modèle final
    if len(scores) >= num_episodes:
        final_path = f'checkpoints/cnn_dqn_model_final_{timestamp}.pth'
        torch.save(agent.qnetwork_local.state_dict(), final_path)
        print(f"\nEntraînement terminé après {num_episodes} épisodes.")
        print(f"Modèle final sauvegardé: {final_path}")
        
        # Tracer et sauvegarder les graphiques finaux
        plot_scores(scores, losses, f'figures/cnn_dqn_training_final_{timestamp}.png')
    
    # Afficher le temps total d'entraînement
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    print(f"Temps total d'entraînement: {hours}h {minutes}m {seconds}s")
    
    return scores, losses

def plot_scores(scores, losses=None, filename=None):
    """Affiche et sauvegarde un graphique des scores et optionnellement des pertes pendant l'entraînement."""
    # Créer la figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))
    
    # Tracer les scores
    axes[0].plot(np.arange(len(scores)), scores, alpha=0.6)
    axes[0].set_ylabel('Score')
    axes[0].set_xlabel('Épisode')
    axes[0].set_title('Scores pendant l\'entraînement CNN-DQN')
    
    # Calculer la moyenne mobile sur 100 épisodes (ou moins si moins d'épisodes)
    window_size = min(100, len(scores))
    if window_size > 0:
        moving_avg = np.convolve(scores, np.ones((window_size,))/window_size, mode='valid')
        axes[0].plot(np.arange(len(moving_avg)) + window_size-1, moving_avg, color='red')
        axes[0].axhline(y=15, color='green', linestyle='--', alpha=0.3, label='Objectif')
        axes[0].legend(['Scores', f'Moyenne mobile ({window_size} épisodes)', 'Objectif'])
    
    # Si des pertes sont disponibles, les tracer aussi
    if losses is not None and len(losses) > 0:
        axes[1].plot(np.arange(len(losses)), losses, alpha=0.6)
        axes[1].set_ylabel('Loss')
        axes[1].set_xlabel('Épisode')
        axes[1].set_title('Pertes pendant l\'entraînement CNN-DQN')
        
        # Calculer la moyenne mobile sur 100 épisodes (ou moins si moins d'épisodes)
        window_size = min(100, len(losses))
        if window_size > 0:
            moving_avg_loss = np.convolve(losses, np.ones((window_size,))/window_size, mode='valid')
            axes[1].plot(np.arange(len(moving_avg_loss)) + window_size-1, moving_avg_loss, color='red')
            axes[1].legend(['Pertes', f'Moyenne mobile ({window_size} épisodes)'])
    
    plt.tight_layout()
    
    # Sauvegarder la figure si un nom de fichier est fourni
    if filename:
        plt.savefig(filename)
        print(f"Graphique sauvegardé: {filename}")
    
    # Afficher la figure
    plt.show()

def evaluate_cnn_dqn(env_path, model_path, num_episodes=10, time_scale=1.0, render=True):
    """
    Évalue un agent CNN-DQN entraîné dans l'environnement FoodCollector visuel.
    
    Params
    ======
        env_path (str): Chemin vers l'exécutable Unity
        model_path (str): Chemin vers le modèle entraîné
        num_episodes (int): Nombre d'épisodes d'évaluation
        time_scale (float): Vitesse de simulation
        render (bool): Afficher l'environnement pendant l'évaluation
    """
    # Configurer le canal pour la configuration du moteur Unity
    channel = EngineConfigurationChannel()
    
    # Initialiser l'environnement Unity
    print(f"Démarrage de l'environnement Unity visuel pour évaluation: {env_path}")
    env = UnityEnvironment(file_name=env_path, side_channels=[channel, CustomChannel()], no_graphics=not render)
    
    # Définir la timescale
    channel.set_configuration_parameters(time_scale=time_scale)
    
    # Réinitialiser l'environnement
    env.reset()
    
    # Obtenir le behavior name par défaut
    behavior_name = list(env.behavior_specs)[0]
    print(f"Behavior name: {behavior_name}")
    
    # Obtenir les spécifications du comportement
    spec = env.behavior_specs[behavior_name]
    
    # Déterminer la taille de l'action
    if spec.action_spec.continuous_size > 0:
        discrete_actions = 3 * 3 * 3  # 3^3 = 27 actions discrètes possibles sans laser
        if len(spec.action_spec.discrete_branches) > 0:
            discrete_actions *= spec.action_spec.discrete_branches[0]  # Multiplier par le nombre d'actions laser
        action_size = discrete_actions
    else:
        discrete_actions = sum(spec.action_spec.discrete_branches)
        action_size = discrete_actions
    
    # Déterminer la forme de l'observation visuelle
    state_shape = None
    
    # Détecter automatiquement la forme des observations
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    if len(decision_steps) > 0:
        agent_id = decision_steps.agent_id[0]
        
        # Trouver les observations visuelles (de forme 3D)
        for i, obs in enumerate(decision_steps[agent_id].obs):
            if len(obs.shape) == 3:
                state_shape = obs.shape
                print(f"Observation visuelle détectée: forme {state_shape}")
                break
    
    if state_shape is None:
        raise ValueError("Aucune observation visuelle n'a été détectée. Assurez-vous que l'environnement fournit des observations visuelles.")
    
    # Initialiser l'agent CNN-DQN
    agent = CNNDQNAgent(state_shape=state_shape, action_size=action_size, seed=42)
    
    # Charger le modèle entraîné
    print(f"Chargement du modèle: {model_path}")
    agent.qnetwork_local.load_state_dict(torch.load(model_path, map_location=device))
    agent.qnetwork_local.eval()  # Passer en mode évaluation
    
    # Liste pour stocker les scores
    scores = []
    
    # Évaluer l'agent
    for i_episode in range(1, num_episodes+1):
        env.reset()
        
        # Obtenir l'état initial
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        agent_ids = decision_steps.agent_id
        
        if len(agent_ids) == 0:
            print("Aucun agent disponible. Vérifiez l'environnement Unity.")
            break
        
        # Pour simplifier, nous évaluons le premier agent
        agent_id = agent_ids[0]
        
        # Extraire l'observation visuelle
        for i, obs in enumerate(decision_steps[agent_id].obs):
            if len(obs.shape) == 3:
                state = obs
                break
        
        # Initialiser le score pour cet épisode
        score = 0
        
        # Exécuter l'épisode
        done = False
        while not done:
            # Sélectionner une action (greedy)
            action_idx = agent.act(state, eps=0.0)
            
            # Convertir l'action discrète en action compatible avec l'environnement
            if spec.action_spec.continuous_size > 0:
                # Préparation des actions pour tous les agents
                continuous_actions = np.zeros((len(agent_ids), spec.action_spec.continuous_size))
                
                # On garde la possibilité d'avoir des actions discrètes au cas où l'environnement en a encore
                if len(spec.action_spec.discrete_branches) > 0:
                    discrete_actions_array = np.zeros((len(agent_ids), len(spec.action_spec.discrete_branches)), dtype=np.int32)
                
                # Configuration de l'action pour l'agent principal
                move_forward = (action_idx // 9) - 1  # -1, 0, 1
                move_side = ((action_idx // 3) % 3) - 1  # -1, 0, 1
                rotate = (action_idx % 3) - 1  # -1, 0, 1
                
                # Appliquer la même action à tous les agents pour simplifier
                for i in range(len(agent_ids)):
                    continuous_actions[i, 0] = move_forward
                    continuous_actions[i, 1] = move_side
                    continuous_actions[i, 2] = rotate
                
                # Créer les actions en utilisant la méthode correcte
                action_tuple = ActionTuple()
                action_tuple.add_continuous(continuous_actions)
                
                # Ajouter les actions discrètes seulement si l'environnement en a
                if len(spec.action_spec.discrete_branches) > 0:
                    # Si l'action est assez grande pour inclure l'action discrète (laser)
                    if action_idx >= 27:
                        use_laser = 1  # Activer le laser
                    else:
                        use_laser = 0  # Ne pas utiliser le laser
                    
                    # Appliquer l'action discrète à tous les agents
                    for i in range(len(agent_ids)):
                        discrete_actions_array[i, 0] = use_laser
                    
                    # Ajouter les actions discrètes au tuple d'action
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
            if agent_id in terminal_steps:
                # L'agent a terminé
                reward = terminal_steps[agent_id].reward
                done = True
            elif agent_id in decision_steps:
                # L'agent continue
                reward = decision_steps[agent_id].reward
                
                # Extraire l'observation visuelle pour la prochaine étape
                for i, obs in enumerate(decision_steps[agent_id].obs):
                    if len(obs.shape) == 3:
                        state = obs
                        break
            else:
                # L'agent n'est pas disponible (peut-être réinitialisé)
                done = True
            
            # Mettre à jour le score
            score += reward
        
        # Sauvegarder le score
        scores.append(score)
        
        # Afficher le score
        print(f'Épisode {i_episode}/{num_episodes}\tScore: {score:.2f}')
    
    # Fermer l'environnement
    env.close()
    
    # Afficher le score moyen
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    print(f"Score moyen sur {num_episodes} épisodes: {mean_score:.2f} ± {std_score:.2f}")
    
    return scores

if __name__ == "__main__":
    # Chemin vers l'exécutable Unity
    env_path = "../FoodCollectorVisual/FoodCollector.exe"
    
    # Mode d'exécution: 'train' pour entraîner un nouveau modèle ou 'eval' pour évaluer un modèle existant
    mode = 'train'
    
    if mode == 'train':
        # Entraîner l'agent
        scores, losses = train_cnn_dqn(
            env_path=env_path,
            num_episodes=2000,
            time_scale=5.0,
            checkpoint_every=100
        )
        
        # Tracer les résultats finaux
        plot_scores(scores, losses)
    elif mode == 'eval':
        # Chemin vers le modèle à évaluer
        model_path = "checkpoints/cnn_dqn_model_final_20250324_000000.pth"  # À remplacer par le chemin réel
        
        # Évaluer l'agent
        scores = evaluate_cnn_dqn(
            env_path=env_path,
            model_path=model_path,
            num_episodes=10,
            time_scale=1.0,  # Vitesse normale pour la visualisation
            render=True  # Afficher l'environnement pendant l'évaluation
        ) 