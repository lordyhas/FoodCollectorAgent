# Implémentation DQN pour FoodCollector (Unity ML-Agents)

Ce projet implémente l'algorithme Deep Q-Network (DQN) pour entraîner un agent dans l'environnement FoodCollector de Unity ML-Agents. L'objectif est de démontrer comment utiliser un algorithme DQN, qui n'est pas nativement supporté par ML-Agents, pour résoudre un problème d'apprentissage par renforcement dans Unity.

## Description de FoodCollector

FoodCollector est un environnement multi-agents où :
- Les agents doivent collecter des sphères vertes (récompense +1)
- Les agents doivent éviter les sphères rouges (pénalité -1)
- Les agents peuvent se déplacer (avant/arrière, gauche/droite, rotation)
- Les agents peuvent utiliser un laser pour interagir avec l'environnement

## Structure du projet

```
.
├── src/
│   ├── dqn_agent.py         # Implémentation de l'agent DQN
│   ├── train_food_collector.py  # Script d'entraînement
│   └── evaluate_model.py    # Script d'évaluation
├── FoodCollector_Windows_x86_64/  # Environnement Unity précompilé
├── requirements.txt         # Dépendances Python
└── README.md                # Ce fichier
```

## Prérequis

- Python 3.7+
- PyTorch
- Unity ML-Agents
- Numpy, Matplotlib

## Installation

1. Créez un environnement virtuel Python (recommandé) :
   ```
   python -m venv venv
   source venv/bin/activate  # Sur Windows: venv\Scripts\activate
   ```

2. Installez les dépendances :
   ```
   pip install -r requirements.txt
   ```

## Utilisation

### Entraînement de l'agent

Pour entraîner un agent DQN dans l'environnement FoodCollector :

```
cd src
python train_food_collector.py
```

L'entraînement :
- Utilise une stratégie epsilon-greedy pour l'exploration
- Sauvegarde des points de contrôle du modèle tous les 100 épisodes
- S'arrête lorsque le score moyen sur les 100 derniers épisodes atteint l'objectif (par défaut : 10)
- Affiche un graphique des scores à la fin de l'entraînement

### Évaluation du modèle

Pour évaluer un modèle déjà entraîné :

```
cd src
python evaluate_model.py
```

Ce script :
- Charge le modèle entraîné (ou le dernier point de contrôle disponible)
- Exécute l'agent pour 5 épisodes sans exploration (epsilon = 0)
- Affiche les scores par épisode et le score moyen

## Adaptation à d'autres environnements

Cette implémentation peut être adaptée à d'autres environnements Unity ML-Agents. Les principales modifications à apporter seraient :

1. Adapter la logique de discrétisation des actions continues si nécessaire
2. Ajuster les hyperparamètres du DQN selon les besoins de l'environnement
3. Modifier la structure du réseau de neurones pour mieux correspondre à la complexité de l'environnement

## Fonctionnalités avancées

L'implémentation actuelle utilise:
- Deep Q-Network (DQN) standard avec replay buffer
- Soft updates pour le réseau cible
- Discrétisation des actions continues pour compatibilité avec DQN

## Améliorations possibles

- Implémentation de variantes comme Double DQN, Dueling DQN, ou Prioritized Experience Replay
- Support natif des actions continues avec DDPG (Deep Deterministic Policy Gradient)
- Optimisation des hyperparamètres avec une recherche par grille ou aléatoire
- Visualisation des Q-values pendant l'entraînement pour la compréhension
- Implémentation d'agents parallèles pour accélérer l'entraînement

## Pourquoi DQN pour FoodCollector?

Bien que Unity ML-Agents prenne en charge nativement PPO et SAC, l'implémentation de DQN :
1. Démontre la flexibilité de l'API ML-Agents
2. Peut être plus adaptée pour certains types de problèmes à action discrète
3. Est généralement plus simple à comprendre conceptuellement que les algorithmes policy-based
4. Sert de base pédagogique pour comprendre l'apprentissage par renforcement profond 