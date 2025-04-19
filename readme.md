# Curiosity‑Driven AI

[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)  
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)  
[![Build Status](https://img.shields.io/badge/CI-passing-brightgreen)](https://github.com/ton-org/curiosity-driven-ai/actions)  

---

## Table des matières

1. [Contexte & Motivation](#contexte--motivation)  
2. [Objectifs du Projet](#objectifs-du-projet)  
3. [Architecture Globale](#architecture-globale)  
4. [Principaux Modules](#principaux-modules)  
5. [Fonctionnement du Méta‑Contrôleur](#fonctionnement-du-méta‑contrôleur)  
6. [Technologies & Dépendances](#technologies--dépendances)  
7. [Installation](#installation)  
8. [Configuration](#configuration)  
9. [Usage & Exemples](#usage--exemples)  
10. [Structure du Projet](#structure-du-projet)  
11. [Tests & Qualité](#tests--qualité)  
12. [Contribution](#contribution)  
13. [Roadmap & Documentation](#roadmap--documentation)  
14. [Licence](#licence)  

---

## Contexte & Motivation

Les IA classiques apprennent à optimiser un objectif externe : classification, génération de texte ou d’images. Mais sans **intention propre**, elles manquent de curiosité, d’autonomie et peinent à généraliser entre domaines.  
**Curiosity‑Driven AI** propose un nouveau paradigme : doter l’agent d’une **motivation intrinsèque**, d’un **méta‑contrôleur** et de modules cognitifs (curiosité, planification, mémoire, analogies…) pour qu’il :

- Se fixe ses propres **buts** et les poursuit  
- Explore et apprend de façon **naturelle** et **continue**  
- **Transfère** ses acquis entre disciplines  

Nous nous appuyons sur les avancées : AlphaGo Zero (MCTS), Transformers, reinforcement learning intrinsèque (ICM, RND), modèles de diffusion (diffu‑GRPO), continual learning, mythèmes (XNets) et apprentissage multimodal.

---

## Objectifs du Projet

1. **Intention** : modéliser un processus de but interne, inspiré des architectures BDI/ACT‑R et des agents autoteliques (IMGEP).  
2. **Curiosité intrinsèque** : ICM & RND pour pousser l’agent vers l’inconnu.  
3. **Planification** : MCTransformer / Tree‑of‑Thoughts pour la réflexion multi‑étapes.  
4. **Apprentissage continu** : adaptateurs (LoRA), replay, EWC pour éviter l’oubli.  
5. **Transdisciplinarité** : XNets pour détecter des mythèmes (isomorphismes conceptuels).  
6. **Mémoire** : répétition espacée SM‑2 pour ancrer les connaissances.  
7. **Évolution dynamique** : ingestion Web via Selenium, mise à jour online.  
8. **Interface** : API FastAPI/WebSocket pour le pilotage en temps réel.  
9. **Suivi** : dashboards matplotlib/seaborn pour visualiser progrès et intentions.

---

## Architecture Globale

┌────────────────────────────────────────────────────┐ │ Méta‑Contrôleur │ │ (active modules selon confiance & complexité) │ └────────────────────────────────────────────────────┘ │ │ │ │ ▼ ▼ ▼ ▼ ┌───────────┐ ┌─────────────────┐ ┌────────────┐ │ Transformer│ │ Curiosité (ICM,│ │ Planif. │ │ (SFT) │ │ RND, IMGEP) │ │ MCTS+ToT │ └───────────┘ └────────────────┘ └────────────┘ │ │ │ ▼ ▼ ▼ ┌────────────────────────────────────────────────────┐ │ Modules Mémoire & Transfert │ │ (Spaced Rep., Continual Learning, XNets) │ └────────────────────────────────────────────────────┘ │ ▼ ┌────────────────────────────────────────────────────┐ │ Ingestion & Enrichissement │ │ (Selenium Web Scraping) │ └────────────────────────────────────────────────────┘ │ ▼ ┌────────────────────────────────────────────────────┐ │ Interaction & Monitoring │ │ (API WebSocket/REST, Dashboards) │ └────────────────────────────────────────────────────┘
## Principaux Modules

1. **SFT (Supervised Fine‑Tuning)**  
   - Initie l’intuition de raisonnement sur GSM8K, MATH, etc.  
2. **Transformer²**  
   - Architecture en cascade : un second Transformer affine la sortie du premier.  
3. **Curiosité Intrinsèque**  
   - ICM (Pathak et al. 2017) + RND pour reward de surprise.  
   - IMGEP pour génération autonome de buts.  
4. **Planification MCTS+Transformer**  
   - MCTransformer ou Tree‑of‑Thoughts pour explorer plusieurs chaînes de pensée avant réponse.  
5. **Spaced Repetition**  
   - Algorithme SM‑2 pour scheduler des révisions sur BabyLM (10 M → 100 M mots).  
6. **Continual Learning**  
   - Adaptateurs (LoRA), replay buffer, EWC, Transformer‑XL online pour apprentissage en flux continu.  
7. **XNets & Mythèmes**  
   - Détection d’isomorphismes conceptuels entre disciplines via blocs linéaires + Softplus.  
8. **diffu‑GRPO**  
   - RL pour modèles de diffusion masquée (one‑step policy gradient + prompt masking).  
9. **Ingestion Web**  
   - Selenium + webdriver‑manager pour récupérer des informations en ligne.  
10. **API Temps Réel**  
    - FastAPI + WebSocket pour envoyer prompts, recevoir états, ajuster hyperparamètres.  
11. **Dashboards**  
    - matplotlib & seaborn pour visualiser métriques, confiance, récompenses, roadmap.

---

## Fonctionnement du Méta‑Contrôleur

Le méta‑contrôleur (`orchestration/controller.py`) inspecte en continu :

- **Confiance** du modèle (entropie, variance)  
- **Complexité** du prompt (longueur, motifs détectés)  
- **Flux de données** (nouvelles entrées, résultat d’ICM)  

Selon des seuils configurables (`configs/network.yaml`) il active :

1. **Transformer** seul  
2. **Curiosité (ICM/RND)**  
3. **Transformer²**  
4. **MCTransformer / ToT**  
5. **Continual Learning**  

Il consigne chaque « état → action → récompense » dans un **blackboard** partagé pour analyse et replay.

---

## Technologies & Dépendances

- **Langage :** Python 3.10+  
- **Modélisation & RL :** PyTorch, Stable‑Baselines3, Transformers  
- **Diffusion RL :** implémentation custom diffu‑GRPO  
- **Planification :** MCTS, custom MCTransformer  
- **Continual Learning :** Avalanche (ou accelerate), LoRA, EWC  
- **Web Scraping :** Selenium, webdriver‑manager  
- **API & Temps Réel :** FastAPI, Uvicorn, WebSockets, Pydantic  
- **Visualisation :** matplotlib, seaborn, TensorBoard, W&B  
- **Qualité & CI :** black, flake8, markdownlint, GitHub Actions  
- **Versioning Données :** DVC / Git LFS  

---

## Installation

```bash
# Cloner le dépôt
git clone https://github.com/ton‑org/curiosity‑driven‑ai.git
cd curiosity‑driven‑ai

# Environnement virtuel
python3.10 -m venv .venv
source .venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt

### Configuration

Adapt the following configuration files before each training run:

* **configs/sft_config.yaml** (Supervised Fine-Tuning):
  ```yaml
  # Example optimized config for RTX4050
  data_pattern: data/sft_examples/*.jsonl
  field: text
  model_name: gpt2-medium        # 345M parameters
  batch_size: 4                  # actual batch size
  gradient_accumulation_steps: 4 # equivalent to batch_size 16
  epochs: 3                      # multiple passes to converge
  learning_rate: 3e-5            # lower LR for GPT2-medium
  max_length: 512
  logging_steps: 20
  evaluation_strategy: steps
  eval_steps: 100
  save_steps: 200
  warmup_steps: 200
  weight_decay: 0.01
  seed: 42
  fp16: true                     # half-precision on GPU
  gradient_checkpointing: true   # reduce activation memory
  ```

* **configs/network.yaml**: thresholds for trust/complexity, hyperparameters for Transformer², XNets.
* **configs/grpo_config.yaml**: masking rate, reward weights for diffu‑GRPO.

* **.env** (optional): Selenium credentials, API tokens.

Adaptez ces fichiers selon votre matériel et vos jeux de données.

1. Entraînement SFT

python scripts/train_sft.py \
  --config configs/sft_config.yaml \
  --output models/sft_finetuned/

2. Curiosité Intrinsèque

python scripts/train_icm.py \
  --env labyrinthe_toy \
  --output models/icm/

3. Transformer²

python scripts/train_mcts_tf.py \
  --config configs/mcts_config.yaml \
  --mode transformer_squared \
  --output models/transformer_squared/

4. diffu‑GRPO

python scripts/train_diffu_grpo.py \
  --config configs/grpo_config.yaml \
  --model models/sft_finetuned/latest.pt \
  --output models/diffu_grpo/

5. API Temps Réel

uvicorn realtime.server:app --reload
# Puis, dans client_example.py :
python realtime/client_example.py

6. Rapport Hebdomadaire

python scripts/report_weekly.py
# génère un rapport Markdown/HTML dans visualization/reports/

Structure du Projet

curiosity_ai_project/
│
├── data/                    # jeux de données segmentés par phase
├── models/
│   ├── diffusion_base/
│   ├── sft_finetuned/
│   └── diffu_grpo/
├── modules/
│   ├── icm/
│   ├── rnd/
│   ├── xnet/
│   ├── spaced_repetition/
│   ├── continual_learning/
│   └── transformer_squared/
├── orchestration/
│   ├── controller.py
│   └── mcts_transformer/
├── ingestion/
│   └── web/
├── realtime/
│   ├── server.py
│   ├── api.py
│   └── client_example.py
├── visualization/
│   └── reports/
├── scripts/
│   ├── train_*.py
│   └── report_weekly.py
├── configs/
│   └── *.yaml
├── docs/
├── .cursor/rules/
├── README.md
├── ROADMAP.md
└── requirements.txt

Tests & Qualité

    Unitaires : pytest tests/unit/

    Intégration : pytest tests/integration/

    Lint : black --check ., flake8 ., markdownlint docs/

    CI : GitHub Actions exécute l’ensemble à chaque PR.

Contribution

    Forker le dépôt & créer une branche thématique.

    Respecter les Cursor Rules (structure, docstrings, mise à jour README/ROADMAP).

    Ajouter tests unitaires pour tout nouveau code.

    Soumettre un Pull Request détaillant vos changements.

Voir CONTRIBUTING.md pour plus de détails.
Roadmap & Documentation

    La Roadmap détaillée se trouve dans ROADMAP.md.

    Consultez docs/network_design.md pour les schémas d’architecture et docs/api_examples.md pour les exemples d’appels.

    Les règles d’édition sont dans .cursor/rules/.

