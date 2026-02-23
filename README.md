# RLkart 🏎️ 🤖

RLkart est une plateforme de simulation de course de karts propulsée par l'apprentissage par renforcement (Reinforcement Learning). Le projet combine un moteur physique (PyBullet), des algorithmes d'IA (Stable Baselines3 & PPO) et une interface web interactive pour concevoir des circuits personnalisés.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.128+-009688.svg)
![Gymnasium](https://img.shields.io/badge/Gymnasium-1.1-28a745.svg)
![PyBullet](https://img.shields.io/badge/PyBullet-3.2+-ff69b4.svg)

## 🌟 Fonctionnalités

- **Apprentissage par Renforcement** : Entraînement d'agents via l'algorithme PPO (Proximal Policy Optimization).
- **Éditeur de Circuits Web** : Interface en JS permettant de dessiner des tracés complexes par splines Catmull-Rom.
- **Simulation Physique** : Utilisation de PyBullet pour une gestion réaliste des collisions, de l'accélération et de la friction.
- **API** : Simulation calculée côté serveur et visualisée de manière fluide côté client.

## 🏗️ Architecture du Projet

```text
RLkart/
├── api.py                 # Serveur FastAPI et gestion des requêtes
├── APISimulator.py        # Simulateur optimisé pour le mode "headless" (API)
├── BaseSimulator.py       # Classe mère de la simulation (Gymnasium Env)
├── Car.py                 # Logique physique et modèles de voitures (Manual/RL)
├── GenTrack.py            # Moteur de génération de circuits et de splines
├── RLModels.py            # Gestionnaire d'entraînement et de chargement SB3
├── TrainSimulator.py      # Script principal d'entraînement
├── TestSimulator.py       # Script de test local avec interface PyBullet
├── Models/                # Modèles PPO entraînés (.zip & .pkl)
└── frontend/              # Interface utilisateur web (HTML/JS)
```

## 🚀 Installation

### 1. Cloner le projet
```bash
git clone https://github.com/Nchpg/RLkart.git
cd RLkart
```

### 2. Créer un environnement virtuel
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Installer les dépendances
```bash
pip install fastapi uvicorn pybullet gymnasium stable-baselines3 shimmy numpy torch
```

## 🎮 Utilisation

### Lancer le Viewer Interactif (Web)
Démarrez le serveur API :
```bash
python api.py
```
Ouvrez ensuite votre navigateur sur [http://localhost:8000](http://localhost:8000).

1. **Dessinez** votre circuit en cliquant sur le canvas (min. 3 points).
3. **Simulez** pour voir l'IA parcourir votre création en temps réel sur le web.

### Lancer un Test Local (GUI PyBullet)
Pour voir l'IA rouler directement dans le moteur PyBullet :
```bash
python TestSimulator.py
```

### Entraîner un modèle
Pour lancer un nouvel entraînement sur des circuits aléatoires :
```bash
python TrainSimulator.py
```

## 🧠 Détails de l'IA

L'agent utilise un vecteur d'observation comprenant :
- La vitesse locale (longitudinale et latérale).
- La vitesse angulaire.
- L'écart par rapport à la ligne centrale.
- 8 rayons de proximité (Lidar) pour détecter les limites de la piste.
- Une anticipation du virage à venir (look-ahead angles à 10, 20 et 40 points).

La fonction de récompense privilégie la progression sur le circuit tout en pénalisant les sorties de piste, l'instabilité directionnelle (shaking) et en encourageant une vitesse élevée.

## 🛠️ Technologies
- **Backend** : Python 3.9+, FastAPI, Uvicorn.
- **IA** : Stable Baselines3 (PPO), Gymnasium.
- **Physique** : PyBullet.
- **Frontend** : HTML5 Canvas, JavaScript (ES6).