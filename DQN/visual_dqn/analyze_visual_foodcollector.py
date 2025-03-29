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
from datetime import datetime


class CustomChannel(SideChannel):
    def __init__(self):
        super().__init__(uuid.UUID("a1d8f7b7-cec8-50f9-b78b-d3e165a78520"))

    def on_message_received(self, msg: bytes) -> None:
        pass


def analyze_environment(env_path, episodes=5, steps_per_episode=100, graphics=False, time_scale=1.0, save_samples=1):
    """
    Analyse l'environnement FoodCollector et affiche un résumé des informations importantes.

    Params:
    ======
        env_path (str): Chemin vers l'exécutable Unity
        episodes (int): Nombre d'épisodes à exécuter pour l'analyse
        steps_per_episode (int): Nombre d'étapes par épisode
        graphics (bool): Afficher les graphiques ou non
        time_scale (float): Vitesse de simulation
        save_samples (int): Nombre d'échantillons visuels à sauvegarder
    """
    print("\n" + "=" * 50)
    print("ANALYSE DE L'ENVIRONNEMENT FOOD COLLECTOR")
    print("=" * 50 + "\n")

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
    print("\n" + "-" * 50)
    print("SPÉCIFICATIONS DES ACTIONS")
    print("-" * 50)

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
    print("\n" + "-" * 50)
    print("SPÉCIFICATIONS DES OBSERVATIONS")
    print("-" * 50)

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

    # Test des observations visuelles si elles existent
    has_visual_obs = "image" in obs_types

    if has_visual_obs and (graphics or save_samples > 0):
        print("\n" + "-" * 50)
        print("TEST DES OBSERVATIONS VISUELLES")
        print("-" * 50)

        print(f"Capture de {max(1, save_samples)} échantillon(s) visuel(s) pour vérification...")

        # Créer le dossier de sauvegarde
        if save_samples > 0 and not os.path.exists('visual_samples'):
            os.makedirs('visual_samples')

        # Boucle pour capturer plusieurs échantillons
        for sample_idx in range(max(1, save_samples)):
            env.reset()

            # Faire quelques pas aléatoires pour avoir des observations différentes
            for _ in range(sample_idx * 2):
                # Prendre une action aléatoire
                if spec.action_spec.continuous_size > 0:
                    # Actions continues aléatoires
                    continuous_actions = np.random.uniform(-1, 1, (1, spec.action_spec.continuous_size))
                    action_tuple = ActionTuple()
                    action_tuple.add_continuous(continuous_actions)

                    # Ajouter des actions discrètes si nécessaire
                    if len(spec.action_spec.discrete_branches) > 0:
                        discrete_actions_array = np.zeros((1, len(spec.action_spec.discrete_branches)), dtype=np.int32)
                        for i, branch_size in enumerate(spec.action_spec.discrete_branches):
                            discrete_actions_array[:, i] = np.random.randint(0, branch_size, 1)
                        action_tuple.add_discrete(discrete_actions_array)

                    env.set_actions(behavior_name, action_tuple)
                    env.step()

            # Obtenir l'observation
            decision_steps, terminal_steps = env.get_steps(behavior_name)

            if len(decision_steps) == 0:
                print("Aucun agent disponible pour capturer des observations visuelles.")
                break

            agent_id = decision_steps.agent_id[0]
            visual_obs_found = False

            for i, obs in enumerate(decision_steps[agent_id].obs):
                if len(obs.shape) == 3:  # C'est une observation visuelle
                    visual_obs_found = True
                    print(f"Visualisation de l'échantillon {sample_idx + 1}/{max(1, save_samples)}, observation {i}:")

                    # Normaliser l'image pour l'affichage
                    image = obs

                    # Obtenir les dimensions
                    if len(image.shape) == 3:
                        # Corriger l'interprétation des dimensions
                        h, w, c = 0, 0, 0

                        # Déterminer l'organisation des dimensions (canaux, hauteur, largeur)
                        if image.shape[0] < image.shape[1] and image.shape[0] < image.shape[2]:
                            # Format (canaux, hauteur, largeur) - le plus probable pour (5, 40, 40)
                            channels = image.shape[0]
                            height = image.shape[1]
                            width = image.shape[2]
                            print(f"  ├─ Format de l'image: {channels} canaux de {height}x{width} pixels")
                        elif image.shape[2] < image.shape[0] and image.shape[2] < image.shape[1]:
                            # Format (hauteur, largeur, canaux) - le format habituel pour matplotlib
                            channels = image.shape[2]
                            height = image.shape[0]
                            width = image.shape[1]
                            print(f"  ├─ Format de l'image: {height}x{width} pixels avec {channels} canaux")
                        else:
                            # Cas indéterminé, considérer comme (hauteur, largeur, canaux)
                            channels = image.shape[2]
                            height = image.shape[0]
                            width = image.shape[1]
                            print(f"  ├─ Format indéterminé, supposant {height}x{width} pixels avec {channels} canaux")

                        # Vérifier si l'image a un nombre non standard de canaux
                        if channels > 4:
                            print(f"  ├─ Image à {channels} canaux (non standard)")

                            # Pour les affichages, nous devons réorganiser les dimensions si elles ne sont pas
                            # au format attendu par matplotlib (hauteur, largeur, canaux)
                            if image.shape[0] == channels:  # Si première dimension = canaux
                                # Convertir de (C, H, W) à (H, W, C) pour l'affichage
                                image_for_display = np.transpose(image, (1, 2, 0))
                                print(f"  ├─ Transposition de {image.shape} à {image_for_display.shape} pour affichage")
                            else:
                                image_for_display = image

                            # Afficher les 3 premiers canaux comme une image RGB si disponibles
                            if channels >= 3 and graphics:
                                # Créer une image RGB avec les 3 premiers canaux
                                if image.shape[0] == channels:  # Format (C, H, W)
                                    # Prendre les 3 premiers canaux et transposer
                                    rgb_image = np.transpose(image[:3], (1, 2, 0))
                                else:  # Format (H, W, C)
                                    rgb_image = image[:, :, :3]

                                # Normaliser les valeurs pour l'affichage
                                if rgb_image.max() > 1.0:
                                    rgb_image = rgb_image / 255.0

                                plt.figure(figsize=(8, 8))
                                plt.imshow(rgb_image)
                                plt.title(f"Échantillon {sample_idx + 1} - RGB (3 premiers canaux)")
                                plt.axis('off')
                                plt.show()

                            # Afficher chaque canal séparément
                            if graphics:
                                fig, axes = plt.subplots(1, channels, figsize=(channels * 3, 4))

                                for ch in range(channels):
                                    # Extraire un canal
                                    if image.shape[0] == channels:  # Format (C, H, W)
                                        channel = image[ch]
                                    else:  # Format (H, W, C)
                                        channel = image[:, :, ch]

                                    # Normaliser pour affichage
                                    if channel.max() > 1.0:
                                        channel = channel / 255.0

                                    # Afficher le canal
                                    if channels > 1:
                                        ax = axes[ch]
                                    else:
                                        ax = axes  # Si un seul canal

                                    ax.imshow(channel, cmap='gray')
                                    ax.set_title(f"Canal {ch}")
                                    ax.axis('off')

                                plt.tight_layout()
                                plt.suptitle(f"Échantillon {sample_idx + 1} - Canaux individuels")
                                plt.show()

                        elif channels == 3:  # RGB standard
                            # Format standard RGB
                            if image.shape[0] == channels:  # Format (C, H, W)
                                # Transposer pour matplotlib (H, W, C)
                                display_image = np.transpose(image, (1, 2, 0))
                            else:
                                display_image = image

                            # Normaliser les valeurs pour l'affichage
                            if display_image.max() > 1.0:
                                display_image = display_image / 255.0

                            # Afficher l'image si graphics est activé
                            if graphics:
                                plt.figure(figsize=(8, 8))
                                plt.imshow(display_image)
                                plt.title(f"Échantillon {sample_idx + 1} - Observation {i} (RGB)")
                                plt.axis('off')
                                plt.show()

                        elif channels == 1:  # Niveau de gris
                            # Image en niveaux de gris
                            if image.shape[0] == channels:  # Format (C, H, W)
                                display_image = image[0]  # Prendre le seul canal
                            else:
                                display_image = np.squeeze(image)  # Enlever la dimension du canal

                            # Normaliser les valeurs pour l'affichage
                            if display_image.max() > 1.0:
                                display_image = display_image / 255.0

                            # Afficher l'image si graphics est activé
                            if graphics:
                                plt.figure(figsize=(8, 8))
                                plt.imshow(display_image, cmap='gray')
                                plt.title(f"Échantillon {sample_idx + 1} - Observation {i} (Niveau de gris)")
                                plt.axis('off')
                                plt.show()
                    else:
                        print(f"  ⚠️ Format d'observation visuelle non reconnu: {image.shape}")

                    # Afficher les statistiques sur l'observation
                    print(f"  ├─ Dimensions: {obs.shape}")
                    print(f"  ├─ Type: {obs.dtype}")
                    print(f"  ├─ Min/Max: {obs.min():.4f}/{obs.max():.4f}")
                    print(f"  ├─ Moyenne/Écart-type: {obs.mean():.4f}/{obs.std():.4f}")

                    # Sauvegarder l'image pour référence
                    if save_samples > 0:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        sample_path = f'visual_samples/sample_{sample_idx + 1}_obs_{i}_{timestamp}'

                        # Sauvegarder le tableau brut au format numpy pour analyse future
                        np.save(f'{sample_path}.npy', obs)

                        # Pour la visualisation, sauvegarder plusieurs formats
                        if len(image.shape) == 3:
                            channels = image.shape[0] if image.shape[0] < image.shape[1] and image.shape[0] < \
                                                         image.shape[2] else image.shape[2]

                            if channels > 3:
                                # Sauvegarder version RGB (3 premiers canaux)
                                if image.shape[0] == channels:  # Format (C, H, W)
                                    rgb_image = np.transpose(image[:3], (1, 2, 0))
                                else:  # Format (H, W, C)
                                    rgb_image = image[:, :, :3]

                                if rgb_image.max() > 1.0:
                                    rgb_image = rgb_image / 255.0

                                plt.figure(figsize=(8, 8))
                                plt.imshow(rgb_image)
                                plt.title(f"Échantillon {sample_idx + 1} - RGB")
                                plt.axis('off')
                                plt.savefig(f'{sample_path}_rgb.png')
                                plt.close()

                                # Sauvegarder chaque canal individuellement
                                for ch in range(channels):
                                    if image.shape[0] == channels:  # Format (C, H, W)
                                        channel = image[ch]
                                    else:  # Format (H, W, C)
                                        channel = image[:, :, ch]

                                    if channel.max() > 1.0:
                                        channel = channel / 255.0

                                    plt.figure(figsize=(8, 8))
                                    plt.imshow(channel, cmap='gray')
                                    plt.title(f"Échantillon {sample_idx + 1} - Canal {ch}")
                                    plt.axis('off')
                                    plt.savefig(f'{sample_path}_canal_{ch}.png')
                                    plt.close()
                            else:
                                # Sauvegarder l'image standard
                                if channels == 3:  # RGB
                                    if image.shape[0] == channels:  # Format (C, H, W)
                                        display_image = np.transpose(image, (1, 2, 0))
                                    else:
                                        display_image = image

                                    if display_image.max() > 1.0:
                                        display_image = display_image / 255.0

                                    plt.figure(figsize=(8, 8))
                                    plt.imshow(display_image)
                                    plt.title(f"Échantillon {sample_idx + 1} - RGB")
                                    plt.axis('off')
                                    plt.savefig(f'{sample_path}.png')
                                    plt.close()
                                elif channels == 1:  # Niveau de gris
                                    if image.shape[0] == channels:  # Format (C, H, W)
                                        display_image = image[0]
                                    else:
                                        display_image = np.squeeze(image)

                                    if display_image.max() > 1.0:
                                        display_image = display_image / 255.0

                                    plt.figure(figsize=(8, 8))
                                    plt.imshow(display_image, cmap='gray')
                                    plt.title(f"Échantillon {sample_idx + 1} - Niveau de gris")
                                    plt.axis('off')
                                    plt.savefig(f'{sample_path}.png')
                                    plt.close()
                        else:
                            # Format inconnu, tenter quand même de sauvegarder
                            plt.figure(figsize=(8, 8))
                            try:
                                plt.imshow(image)
                            except:
                                plt.text(0.5, 0.5, f"Format non affichable: {image.shape}",
                                         ha='center', va='center', fontsize=12)
                            plt.title(f"Échantillon {sample_idx + 1} - Format inconnu")
                            plt.axis('off')
                            plt.savefig(f'{sample_path}.png')

                        # Indiquer où les fichiers ont été sauvegardés
                        if len(image.shape) == 3:
                            channels = image.shape[0] if image.shape[0] < image.shape[1] and image.shape[0] < \
                                                         image.shape[2] else image.shape[2]

                            if channels > 3:
                                print(f"  └─ Images sauvegardées: {sample_path}_rgb.png et {sample_path}_canal_*.png")
                            else:
                                print(f"  └─ Image sauvegardée: {sample_path}.png")

            if not visual_obs_found:
                print("Aucune observation visuelle n'a été trouvée dans les données.")
                print("Vérifiez que l'environnement Unity est configuré pour fournir des observations visuelles.")
                break
    elif has_visual_obs and not (graphics or save_samples > 0):
        print(
            "\n⚠️ Des observations visuelles ont été détectées, mais le mode graphique est désactivé et save_samples=0.")
        print("   Exécutez avec --graphics ou --save_samples > 0 pour visualiser/sauvegarder les observations.")

    # Analyser les récompenses et pénalités en exécutant quelques épisodes
    print("\n" + "-" * 50)
    print("ANALYSE DES RÉCOMPENSES")
    print("-" * 50)

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
                    discrete_actions_array = np.zeros((len(agent_ids), len(spec.action_spec.discrete_branches)),
                                                      dtype=np.int32)
                    for i, branch_size in enumerate(spec.action_spec.discrete_branches):
                        discrete_actions_array[:, i] = np.random.randint(0, branch_size, len(agent_ids))
                    action_tuple.add_discrete(discrete_actions_array)
            else:
                # Uniquement des actions discrètes
                action_tuple = ActionTuple()
                discrete_actions_array = np.zeros((len(agent_ids), len(spec.action_spec.discrete_branches)),
                                                  dtype=np.int32)
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
        print(f"Épisode {episode + 1}/{episodes}: Récompense totale = {episode_reward:.2f}, Étapes = {episode_length}")

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
    print("\n" + "=" * 50)
    print("RÉSUMÉ DES INFORMATIONS CLÉS POUR L'ENTRAÎNEMENT")
    print("=" * 50)

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
    print("\n" + "-" * 50)
    print("RECOMMANDATIONS DE CONFIGURATION")
    print("-" * 50)

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
    parser.add_argument('--visual', action='store_true',
                        help='Utiliser l\'environnement visuel FoodCollectorVisual au lieu de l\'environnement standard')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Nombre d\'épisodes à exécuter pour l\'analyse')
    parser.add_argument('--steps', type=int, default=10,
                        help='Nombre d\'étapes maximum par épisode')
    parser.add_argument('--graphics', action='store_true',
                        help='Afficher les graphiques')
    parser.add_argument('--time_scale', type=float, default=2.0,
                        help='Vitesse de simulation Unity (1.0 = temps réel)')
    parser.add_argument('--save_samples', type=int, default=1,
                        help='Nombre d\'échantillons visuels à sauvegarder (0 pour désactiver)')

    args = parser.parse_args()

    # Utiliser l'environnement visuel si l'option --visual est spécifiée
    if args.visual and not args.env_path:
        env_path = "../FoodCollectorVisual/FoodCollector.exe"
    else:
        env_path = args.env_path

    # Analyser l'environnement
    analyze_environment(
        env_path=env_path,
        episodes=args.episodes,
        steps_per_episode=args.steps,
        graphics=args.graphics,
        time_scale=args.time_scale,
        save_samples=args.save_samples
    )