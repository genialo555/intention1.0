# Curiosity‑Driven AI

[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)  
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)  
[![Build Status](https://img.shields.io/badge/CI-passing-brightgreen)](https://github.com/ton-org/curiosity-driven-ai/actions)  

---

## Table des matières

1. [Vision Initiale et Objectifs Fondamentaux](#vision-initiale-et-objectifs-fondamentaux)  
2. [Évolution de l'Architecture Technique](#evolution-de-larchitecture-technique)  
3. [Contexte & Motivation](#contexte--motivation)  
4. [Objectifs du Projet](#objectifs-du-projet)  
5. [Architecture Globale](#architecture-globale)  
6. [Principaux Modules](#principaux-modules)  
7. [Fonctionnement du Méta‑Contrôleur](#fonctionnement-du-méta‑contrôleur)  
8. [Technologies & Dépendances](#technologies--dépendances)  
9. [Installation](#installation)  
10. [Configuration](#configuration)  
11. [Usage & Exemples](#usage--exemples)  
12. [Structure du Projet](#structure-du-projet)  
13. [Tests & Qualité](#tests--qualité)  
14. [Contribution](#contribution)  
15. [Roadmap & Documentation](#roadmap--documentation)  
16. [Licence](#licence)  

---

## Vision Initiale et Objectifs Fondamentaux

### 1.1 Inspiration & But Ultime
Le projet « Curiosity-Driven AI » vise à concevoir une intelligence artificielle capable d'apprendre et d'explorer son environnement de la même façon qu'un enfant, guidée par une soif de découverte plutôt que par des seules récompenses extrinsèques. L'ambition est de dépasser les limites de la « Narrow AI » pour tendre vers une forme d'intelligence plus générale, où la curiosité intrinsèque motive l'apprentissage continu et la recherche de nouveaux savoirs.

### 1.2 Mécanisme de Récompense Conceptuel
La première idée fut de créer un système de « cartes virtuelles à collectionner » : chaque découverte ou progrès interne de l'IA générait une nouvelle carte à ajouter à sa collection, fournissant une rétroaction ludique et stimulante pour encourager l'exploration de tâches variées.

### 1.3 Apprentissage Progressif Cible
- **Phase 1** : Résolution de problèmes simples (calculs élémentaires, énigmes logiques) pour établir un socle de compétences structurées.
- **Phase 2** : Acquisition naturelle du langage via l'étude de livres pour enfants en anglais, afin de maîtriser progressivement la compréhension et la génération de texte.
- **Phase 3** : Exploration autonome de concepts plus complexes (sciences, mathématiques, philosophie) grâce à la détection de « mythèmes », c'est-à-dire de structures conceptuelles récurrentes dans différents domaines.

### 1.4 Notion Clé – Mythèmes
Les mythèmes sont des isomorphismes conceptuels – des motifs structurels communs à plusieurs disciplines. Les repérer permet à l'IA de transférer profondément ses acquis d'un domaine à un autre et de développer une « méta-curiosité » : comprendre que découvrir un nouveau concept dans un domaine peut éclairer la résolution de problèmes dans un autre.

---

## Évolution de l'Architecture Technique

### 2.1 Modèles de Base Considérés
- **AlphaGo Zero + MCTS** pour la planification structurée et l'exploration d'arbres de décision.  
- **Transformers** (démarrant sur GPT-2 et envisagés plus tard DeepSeek R1 lite) servant de colonne vertébrale pour le langage, le raisonnement supervisé (SFT) et la génération.  

### 2.2 Apprentissage par Renforcement (RL)
- **Agent Principal** entraîné avec PPO (Stable-Baselines3 v2+) combinant récompenses extrinsèques et intrinsèques.  
- **Exploration pour LMs** : tentative d'adapter un policy gradient (inspiré du diffu-GRPO) aux modèles de type diffusion ou masqués pour améliorer le raisonnement.  

### 2.3 Modules de Curiosité Intrinsèque
- **ICM (Intrinsic Curiosity Module)** : forward model + inverse model ; la récompense intrinsèque = erreur du forward model.  
- **RND (Random Network Distillation)** : réseau cible fixe et aléatoire + prédicteur ; récompense = erreur de prédiction du prédicteur.  

### 2.4 Mémoire et Stabilité
- **Répétition Espacée** (SM-2) pour contrer l'oubli catastrophique en apprentissage continu.  
- **Continual Learning** (EWC, replay, adaptateurs LoRA) afin d'ajouter de nouvelles compétences sans effacer les anciennes.  

### 2.5 Architectures Neuronales Spécifiques
- **Transformer² (CascadeTransformer)** : pipeline coarse-to-fine avec LM figé + LM raffineur entraînable (optimisation LoRA).  
- **XNets** : réseaux contrastifs (MLP + activations dédiées) pour déceler les mythèmes en comparant embeddings de concepts.  

### 2.6 Orchestration et Contrôle
- **Méta-Contrôleur** (`orchestration/controller.py`) qui choisit dynamiquement quel module activer (SFT, ICM/RND, Transformer², MCTS/ToT, diffu-GRPO) selon la confiance et la complexité de la tâche.  
- **Blackboard** : espace clé-valeur pour le partage d'informations (chemins de checkpoints, métriques).  

### 2.7 Enrichissement et Interaction
- **Ingestion Web** (Selenium) pour la recherche dynamique d'informations.  
- **API Temps-Réel** (FastAPI/WebSocket) pour piloter entraînement et inférence, et afficher les métriques en direct.  
- **Visualisation** sur dashboards (Matplotlib/Seaborn, TensorBoard, W&B) pour suivre pertes, récompenses, perplexité et embeddings.  

---

## Contexte & Motivation

Les IA classiques apprennent à optimiser un objectif externe : classification, génération de texte ou d'images. Mais sans **intention propre**, elles manquent de curiosité, d'autonomie et peinent à généraliser entre domaines.  
**Curiosity‑Driven AI** propose un nouveau paradigme : doter l'agent d'une **motivation intrinsèque**, d'un **méta‑contrôleur** et de modules cognitifs (curiosité, planification, mémoire, analogies…) pour qu'il :

- Se fixe ses propres **buts** et les poursuit  
- Explore et apprend de façon **naturelle** et **continue**  
- **Transfère** ses acquis entre disciplines  

Nous nous appuyons sur les avancées : AlphaGo Zero (MCTS), Transformers, reinforcement learning intrinsèque (ICM, RND), modèles de diffusion (diffu‑GRPO), continual learning, mythèmes (XNets) et apprentissage multimodal.

---

## Objectifs du Projet

1. **Intention** : modéliser un processus de but interne, inspiré des architectures BDI/ACT‑R et des agents autoteliques (IMGEP).  
2. **Curiosité intrinsèque** : ICM & RND pour pousser l'agent vers l'inconnu.  
3. **Planification** : MCTransformer / Tree‑of‑Thoughts pour la réflexion multi‑étapes.  
4. **Apprentissage continu** : adaptateurs (LoRA), replay, EWC pour éviter l'oubli.  
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
   - Initie l'intuition de raisonnement sur GSM8K, MATH, etc.  
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
   - Détection d'isomorphismes conceptuels entre disciplines via blocs linéaires + Softplus.  
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
- **Flux de données** (nouvelles entrées, résultat d'ICM)  

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
* **configs/grpo_config.yaml**: masking rate, reward weights for diffu‑GRPO
  ```yaml
  # Default diffu‑GRPO config
  model_path: models/sft_finetuned/latest.pt  # Path to pretrained SFT checkpoint
  mask_rate: 0.15           # Proportion of tokens to mask per step
  reward_weight: 1.0        # Weight for reward computation
  use_log_ratio: false      # Use log-ratio between unmasked/masked loss if true
  timesteps: 1000           # Number of diffusion+policy gradient steps
  logging_steps: 100        # Interval for logging progress
  save_steps: 100           # Steps between checkpoint saves
  learning_rate: 5e-5       # Learning rate for policy optimizer
  seed: 42                  # Random seed
  seq_len: 16               # Sequence length for input
  batch_size: 2             # Batch size during training
  ```
* **configs/icm_config.yaml** (Intrinsic Curiosity Module - ICM):
  ```yaml
  # Default ICM config
  env: CartPole-v1          # Gym environment name
  timesteps: 1000           # Number of timesteps to run the agent
  logging_steps: 100        # Interval for logging progress
  seed: 42                  # Random seed for reproducibility
  lr: 1e-3                  # Learning rate for ICM optimizer
  save_steps: 100           # Steps between model checkpoint saves
  ```
* **configs/rnd_config.yaml** (Random Network Distillation - RND):
  ```yaml
  # Default RND config
  env: CartPole-v1          # Gym environment name
  timesteps: 1000           # Number of timesteps to run the placeholder agent
  logging_steps: 100        # Interval for logging progress
  seed: 42                  # Random seed for reproducibility
  # Add lr, save_steps, and other hyperparameters when implementing full RND module
  ```
* **configs/transformer2_config.yaml** (Transformer² placeholder config):
  ```yaml
  # Transformer² placeholder config
  model_name: gpt2            # Pretrained HF model name
  seq_len: 16                 # Sequence length for input
  batch_size: 2               # Batch size for placeholder training
  timesteps: 10               # Number of training steps on random token data
  learning_rate: 5e-5         # Learning rate for refiner optimizer
  logging_steps: 1            # Log interval
  save_steps: 5               # Checkpoint save interval
  seed: 42                    # Seed for reproducibility
  peft_enable: false          # Enable LoRA adapters via PEFT (requires `peft` package)
  peft_r: 8                   # LoRA rank
  peft_alpha: 32              # LoRA scaling factor
  peft_dropout: 0.05          # LoRA dropout rate
  ```
* **.env** (optionnel) : Selenium credentials, API tokens.

Adaptez ces fichiers selon votre matériel et vos jeux de données.

1. Entraînement SFT

```bash
python scripts/train_sft.py \
  --config configs/sft_config.yaml \
  --output models/sft_finetuned/
```

2. Intrinsic Curiosity Module (ICM)

python scripts/train_icm.py \
  --config configs/icm_config.yaml \
  --output models/icm/

3. Random Network Distillation (RND)

python scripts/train_rnd.py \
  --config configs/rnd_config.yaml \
  --output models/rnd/

4. Transformer²

```bash
python scripts/train_mcts_tf.py \
  --config configs/transformer2_config.yaml \
  --output models/mcts_transformer/
```

5. diffu-GRPO

```bash
python scripts/train_diffu_grpo.py \
  --config configs/grpo_config.yaml \
  --output models/diffu_grpo/
```

6. API Temps Réel

uvicorn realtime.server:app --reload
# Puis, dans client_example.py :
python realtime/client_example.py

7. Rapport Hebdomadaire

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

    CI : GitHub Actions exécute l'ensemble à chaque PR.

Contribution

    Forker le dépôt & créer une branche thématique.

    Respecter les Cursor Rules (structure, docstrings, mise à jour README/ROADMAP).

    Ajouter tests unitaires pour tout nouveau code.

    Soumettre un Pull Request détaillant vos changements.

Voir CONTRIBUTING.md pour plus de détails.
Roadmap & Documentation

    La Roadmap détaillée se trouve dans ROADMAP.md.

    Consultez docs/network_design.md pour les schémas d'architecture et docs/api_examples.md pour les exemples d'appels.

    Les règles d'édition sont dans .cursor/rules/.

## Usage & Exemples

### 5. Pipeline recommandé
Assurez-vous d'avoir configuré votre GPU (voir section « Matériel & Configuration GPU »), puis suivez ces étapes :

1. Data Ingestion & Pré-traitement
   ```bash
   python scripts/prepare_data.py \
     --input data/raw_prompts.jsonl \
     --output data/processed/
   ```
2. Supervised Fine-Tuning (SFT)
   ```bash
   python scripts/train_sft.py \
     --config configs/sft_config.yaml \
     --output models/sft_finetuned/
   ```
3. (Optionnel) Évaluation SFT
   ```bash
   python scripts/evaluate_sft.py \
     --model models/sft_finetuned/latest.pt \
     --data data/processed/val.jsonl
   ```
4. diffu-GRPO (RL à partir du checkpoint SFT)
   ```bash
   python scripts/train_diffu_grpo.py \
     --config configs/grpo_config.yaml \
     --output models/diffu_grpo_from_sft/
   ```
5. Évaluation finale & déploiement
   ```bash
   python scripts/evaluate_diffu_grpo.py \
     --model models/diffu_grpo_from_sft/final_model \
     --data data/processed/test.jsonl
   ```
6. API Temps-Réel & Dashboards
   ```bash
   uvicorn realtime.server:app --reload
   ```
Cette organisation découple clairement chaque phase et permet de réexécuter ou remplacer un bloc sans tout relancer.
### 6. Structure finale du projet
```text
curiosity_ai_project/
├── data/
│   ├── raw_prompts.jsonl
│   └── processed/
├── models/
│   ├── sft_finetuned/
│   └── diffu_grpo_from_sft/
├── scripts/
│   ├── prepare_data.py
│   ├── train_sft.py
│   ├── evaluate_sft.py
   ├── train_diffu_grpo.py
   ├── evaluate_diffu_grpo.py
   └── report_weekly.py
├── configs/
│   ├── sft_config.yaml
│   └── grpo_config.yaml
└── realtime/
    └── server.py
```

## Matériel & Configuration GPU
Le pipeline exploite idéalement une configuration multi-GPU (p.ex. RTX 4050 interne + eGPU RTX 3090).
Avant chaque exécution d'entraînement, choisissez et configurez votre GPU :
```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=<GPU_INDEX>  # 0 pour la 4050, 1 pour la 3090
```
Si le GPU est bloqué : `sudo nvidia-smi --gpu-reset -i <GPU_INDEX>`
Pour forcer le mode performance : `nvidia-smi -pm 1`

### Pipeline détaillé pour GPU RTX 3090

Pour exécuter l'entraînement sur votre eGPU NVIDIA GeForce RTX 3090 (index GPU 1), suivez ces étapes :

1. Configuration du GPU
```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1  # Utiliser la RTX 3090
sudo nvidia-smi --gpu-reset -i 1  # Réinitialiser si nécessaire
nvidia-smi -pm 1                 # Mode performance
```

2. Pré-traitement des données
```bash
python scripts/prepare_data.py \
  --input data/raw_prompts.jsonl \
  --output data/processed/
```

3. Fine-Tuning supervisé (SFT)
```bash
python scripts/train_sft.py \
  --config configs/sft_config.yaml \
  --output models/sft_finetuned/
```

4. Évaluation du SFT
```bash
python scripts/evaluate_sft.py \
  --model models/sft_finetuned/latest.pt \
  --data data/processed/val.jsonl
```

5. Entraînement diffu-GRPO (RL)
```bash
python scripts/train_diffu_grpo.py \
  --config configs/grpo_config.yaml \
  --output models/diffu_grpo_from_sft/
```

6. Évaluation finale du diffu-GRPO
```bash
python scripts/evaluate_diffu_grpo.py \
  --model models/diffu_grpo_from_sft/final_model \
  --data data/processed/test.jsonl
```

7. Lancement de l'API Temps-Réel & Dashboards
```bash
uvicorn realtime.server:app --reload
```

Cette configuration dédiée à la RTX 3090 garantit une gestion et une répartition optimales des ressources GPU.

### Pipeline complet d'entraînement

Pour démarrer de zéro et couvrir toutes les phases (sur n'importe quel GPU), suivez cet enchaînement :

1. Configuration GPU
```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=<GPU_INDEX>  # ex. 1 pour RTX 3090
sudo nvidia-smi --gpu-reset -i <GPU_INDEX>  # reset si bloqué
nvidia-smi -pm 1                         # mode performance
```  

2. Pré-traitement des données
```bash
python scripts/prepare_data.py \
  --input data/raw_prompts.jsonl \
  --output data/processed/
```  

3. Supervised Fine-Tuning (SFT)
```bash
python scripts/train_sft.py \
  --config configs/sft_config.yaml \
  --output models/sft_finetuned/
```  

4. Intrinsic Curiosity Module (ICM) (optionnel)
```bash
python scripts/train_icm.py \
  --config configs/icm_config.yaml \
  --output models/icm/
```  

5. Random Network Distillation (RND) (optionnel)
```bash
python scripts/train_rnd.py \
  --config configs/rnd_config.yaml \
  --output models/rnd/
```  

6. Transformer² (CascadeTransformer) (optionnel)
```bash
python scripts/train_transformer2_real.py \
  --output models/transformer_squared/
```  

7. Planification MCTS / Tree-of-Thoughts (optionnel)
```bash
python scripts/train_mcts_tf.py \
  --config configs/transformer2_config.yaml \
  --output models/mcts_transformer/
```  

8. diffu-GRPO (Renforcement)
```bash
python scripts/train_diffu_grpo.py \
  --config configs/grpo_config.yaml \
  --output models/diffu_grpo_from_sft/
```  

9. Évaluation finale diffu-GRPO
```bash
python scripts/evaluate_diffu_grpo.py \
  --model models/diffu_grpo_from_sft/final_model \
  --data data/processed/test.jsonl
```  

10. API Temps-Réel & Dashboards
```bash
uvicorn realtime.server:app --reload
```  
Cette section regroupe toutes les étapes essentielles pour initialiser et exécuter l'ensemble du pipeline.

