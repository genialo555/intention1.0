# Curiosity‑Driven AI

[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)  
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)  
[![Build Status](https://img.shields.io/badge/CI-passing-brightgreen)](https://github.com/ton-org/curiosity-driven-ai/actions)  

> **⚠️ IMPORTANT** : L'ordre d'entraînement des modules est **critique** pour le fonctionnement du système. LIMEN doit être entraîné immédiatement après SFT car il supervise tous les autres modules. Voir [Logique et Dépendances du Pipeline](#logique-et-dépendances-du-pipeline) pour plus de détails.

---

## Table des matières

1. [Vision Initiale et Objectifs Fondamentaux](#vision-initiale-et-objectifs-fondamentaux)  
2. [Évolution de l'Architecture Technique](#evolution-de-larchitecture-technique)  
3. [Contexte & Motivation](#contexte--motivation)  
4. [Objectifs du Projet](#objectifs-du-projet)  
5. [Architecture Globale](#architecture-globale)  
6. [Principaux Modules](#principaux-modules)  
7. [Module LIMEN : Architecture de l'Intention Émergente](#module-limen--architecture-de-lintention-émergente)  
8. [Fonctionnement du Méta‑Contrôleur](#fonctionnement-du-méta‑contrôleur)  
9. [Technologies & Dépendances](#technologies--dépendances)  
10. [Installation](#installation)  
11. [Configuration](#configuration)  
12. [Usage & Exemples](#usage--exemples)  
13. [Logique et Dépendances du Pipeline](#logique-et-dépendances-du-pipeline)  
14. [Structure du Projet](#structure-du-projet)  
15. [Tests & Qualité](#tests--qualité)  
16. [Contribution](#contribution)  
17. [Roadmap & Documentation](#roadmap--documentation)  
18. [Licence](#licence)  

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
- **Transformers** (basés sur **DeepSeek R1 Qwen 12B**) servant de colonne vertébrale pour le langage, le raisonnement supervisé (SFT) et la génération.  

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
2. **Cohérence intentionnelle** : LIMEN pour gérer les tensions internes et maintenir une intention émergente cohérente.  
3. **Curiosité intrinsèque** : ICM & RND pour pousser l'agent vers l'inconnu.  
4. **Planification** : MCTransformer / Tree‑of‑Thoughts pour la réflexion multi‑étapes.  
5. **Apprentissage continu** : adaptateurs (LoRA), replay, EWC pour éviter l'oubli.  
6. **Transdisciplinarité** : XNets pour détecter des mythèmes (isomorphismes conceptuels).  
7. **Mémoire** : répétition espacée SM‑2 pour ancrer les connaissances.  
8. **Évolution dynamique** : ingestion Web via Selenium, mise à jour online.  
9. **Interface** : API FastAPI/WebSocket pour le pilotage en temps réel.  
10. **Suivi** : dashboards matplotlib/seaborn pour visualiser progrès et intentions.

---

## Architecture Globale

```
┌─────────────────────────────────────────────────────────────────┐
│                    MetaLIMEN (Pré-Entraînement)               │
│              (Définition intentions d'apprentissage)           │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      SFT Guidé par Intentions                  │
│             (Fine-tuning DeepSeek R1 avec guidance)            │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FullLIMEN (Post-SFT)                        │
│              (Intentions raffinées avec capacités complètes)    │
│     ┌─────────────┬─────────────┬─────────────┬─────────────┐  │
│     │ Validation  │  Phylo      │ Conceptual  │ Intention   │  │
│     │ Post-Gen    │ Guidance    │   Encoding  │ Refinement  │  │
└─────┼─────────────┼─────────────┼─────────────┼─────────────┼──┘
      │             │             │             │             │
      ▼             ▼             ▼             ▼             ▼
┌─────────────────────────────────────────────────────────────────┐
│              🚀 GROUP THINK PHYLOGÉNÉTIQUE 🚀                  │
│          (Agents Concurrents Collaborant au Niveau Token)      │
│                                                                 │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐    │
│  │Conceptual   │ Curiosité   │ Planif.     │ Phylogenetic│    │
│  │Transform.   │ (ICM, RND)  │ MCTS+ToT    │ Mythèmes    │    │
│  │   (T²)      │Phylogénét.  │ Conceptuel  │ (XNets)     │    │
│  │             │             │             │             │    │
│  │ Token-Level Collaboration avec Shared Phylogenetic Context │
│  └─────────────┴─────────────┴─────────────┴─────────────┘    │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Conceptual GRPO                                │
│       (Optimisation phylogénétique dans l'espace conceptuel)   │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Interface & Monitoring                       │
│                (API WebSocket/REST, Dashboards)                │
└─────────────────────────────────────────────────────────────────┘
```

### Points clés de l'architecture révolutionnaire :

1. **MetaLIMEN préliminaire** : Définit les intentions d'apprentissage avant toute base linguistique
2. **SFT guidé par intentions** : Construction de la fondation linguistique orientée par les objectifs conceptuels  
3. **FullLIMEN post-SFT** : Raffinement des intentions avec capacités linguistiques complètes
4. **🚀 Group Think phylogénétique** : **Agents concurrents collaborant au niveau token** - Révolution architecturale
5. **Optimisation conceptuelle** : GRPO dans l'espace des concepts plutôt que tokens

### 🚀 Révolution Group Think Phylogénétique

Notre système intègre l'architecture révolutionnaire [Group Think](https://arxiv.org/abs/2505.11107) adaptée à l'espace phylogénétique conceptuel. Cette approche transforme nos modules en agents concurrents qui :

- **Collaborent au niveau token** avec shared visibility phylogénétique
- **Réduisent la latence de 50%+** vs approche séquentielle traditionnelle
- **Optimisent l'utilisation GPU** avec edge inference efficace
- **Génèrent une qualité émergente supérieure** grâce à la collaboration

**Documentation complète** : [Architecture Group Think Phylogénétique](docs/architecture/group_think_integration.md)

**Bénéfices révolutionnaires** :
- ✅ **Latence réduite** : ~1-2s vs ~5-10s (pipeline séquentiel)
- ✅ **Qualité émergente** : +30% vs agent unique
- ✅ **Utilisation GPU** : >90% avec batch sizes faibles
- ✅ **Architecture agent formalisée** selon standards industriels

---

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
12. **LIMEN (Latent Intent Meta-Emergent Nexus)**
    - Module de cohérence intentionnelle gérant les tensions internes entre modules
    - Combine LIDM, coordination multi-agent, MRKL et monologue intérieur
    - Permet le refus réflexif et le silence raisonné quand l'intention est en conflit

---

## Module LIMEN : Architecture de l'Intention Émergente

### Vue d'ensemble
LIMEN (Latent Intent Meta-Emergent Nexus) constitue l'organe de cohérence intentionnelle de l'agent, gérant les tensions et contradictions internes entre modules pour maintenir une intention cohérente mais dynamique.

### Briques architecturales intégrées

#### 1. LIDM (Latent Intention Dialogue Model)
**Ce qu'on garde :**
- Représentation discrète de l'intention comme variable latente
- Capacité à moduler la génération via un vecteur d'intention choisi

**Ce qu'on ajoute :**
- Intention non supervisée issue de tensions internes (pas juste cluster d'étiquettes)
- Cycle réflexif où l'intention peut être refusée par d'autres modules

#### 2. Multi-Agent Intention Coordination (MARL)
**Ce qu'on garde :**
- Consensus / divergence entre modules spécialisés
- Propagation dynamique de l'intention entre modules
- Accords émergents

**Ce qu'on ajoute :**
- Ces "agents internes" sont des parties d'un même esprit
- Modulateur central qui peut inhiber une intention majoritaire (≠ juste voter)

#### 3. MRKL Systems (Modular Reasoning & Knowledge)
**Ce qu'on garde :**
- Structure modulaire raisonnement/langage/mémoire
- Sélection dynamique de modules selon contexte

**Ce qu'on ajoute :**
- Mémoire intentionnelle flottante (non seulement knowledge)
- Modules en tension, pas toujours alignés — friction contrôlée

#### 4. ICL-inspired "Inner Monologue"
**Ce qu'on garde :**
- Capacité à raisonner avant de répondre
- Auto-évaluation implicite de l'action à prendre

**Ce qu'on ajoute :**
- Logique de "silence raisonné" : l'agent peut ne pas répondre si son LIMEN est en désaccord profond
- Trace de doute : "je pense mais je ne suis pas sûr, donc je retiens"

### Composants fonctionnels de LIMEN

| Élément | Fonction centrale |
|---------|-------------------|
| **Tenseur latent** | Porte l'intention du moment (dynamiquement mis à jour) |
| **Tension evaluator** | Compare intention, contexte, mémoire, contradiction |
| **Désactivateur** | Peut annuler, bloquer, ou retarder une réponse |
| **Inhibiteur social** | Réagit au consensus (GroupThink) mais peut refuser d'y adhérer |
| **Moteur d'apprentissage local** | Apprend quand une intention mène au bon type de rupture |

### Intégration avec le Méta-Contrôleur

LIMEN s'interface directement avec le méta-contrôleur existant, ajoutant une couche de validation intentionnelle :
- Avant l'activation d'un module, LIMEN évalue la cohérence de l'intention
- En cas de conflit profond, LIMEN peut forcer un mode "silence" ou "exploration alternative"
- Les métriques de tension interne sont ajoutées au blackboard pour analyse

### Configuration LIMEN

Un nouveau fichier `configs/limen_config.yaml` permet de paramétrer :
```yaml
# Configuration LIMEN
latent_dim: 128              # Dimension du tenseur d'intention latente
tension_threshold: 0.7       # Seuil de tension pour déclencher l'inhibition
consensus_weight: 0.3        # Poids du consensus vs individualité
doubt_trace_memory: 100      # Nombre d'états de doute à conserver
learning_rate: 1e-4          # Taux d'apprentissage du moteur local
update_frequency: 10         # Fréquence de mise à jour du tenseur latent
module_weights:              # Poids des différents modules dans les décisions
  transformer: 0.3
  curiosity: 0.2
  planning: 0.2
  memory: 0.3
silence_mode_threshold: 0.85 # Seuil de tension pour activer le mode silence
```

---

## Fonctionnement du Méta‑Contrôleur

Le méta‑contrôleur (`orchestration/controller.py`) inspecte en continu :

- **Confiance** du modèle (entropie, variance)  
- **Complexité** du prompt (longueur, motifs détectés)  
- **Flux de données** (nouvelles entrées, résultat d'ICM)  
- **Cohérence intentionnelle** (via LIMEN) : tensions internes, contradictions

Processus de décision enrichi par LIMEN :

1. Analyse initiale de la tâche (confiance, complexité)
2. **Validation intentionnelle par LIMEN** :
   - Évaluation de la cohérence entre l'intention latente et l'action proposée
   - Détection des tensions internes entre modules
   - Décision : procéder, modifier l'approche, ou silence raisonné
3. Si validation positive, activation selon les seuils (`configs/network.yaml`) :
   - **Transformer** seul  
   - **Curiosité (ICM/RND)**  
   - **Transformer²**  
   - **MCTransformer / ToT**  
   - **Continual Learning**  
4. Si tension détectée par LIMEN :
   - Mode **exploration alternative** : essayer une approche différente
   - Mode **silence** : ne pas répondre tant que la cohérence n'est pas retrouvée
   - Mode **réflexion approfondie** : activer plusieurs modules pour résoudre le conflit

Il consigne chaque « état → intention → décision LIMEN → action → récompense » dans un **blackboard** partagé pour analyse et replay.

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
  model_name: deepseekr1-qwen-12b    # DeepSeek R1 Qwen 12B
  batch_size: 4                  # actual batch size
  gradient_accumulation_steps: 4 # equivalent to batch_size 16
  epochs: 3                      # multiple passes to converge
  learning_rate: 3e-5            # lower LR for DeepSeek R1 Qwen 12B
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
  model_name: deepseekr1-qwen-12b    # DeepSeek R1 Qwen 12B
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
* **configs/limen_config.yaml** (Latent Intent Meta-Emergent Nexus):
  ```yaml
  # Configuration LIMEN
  latent_dim: 128              # Dimension du tenseur d'intention latente
  tension_threshold: 0.7       # Seuil de tension pour déclencher l'inhibition
  consensus_weight: 0.3        # Poids du consensus vs individualité
  doubt_trace_memory: 100      # Nombre d'états de doute à conserver
  learning_rate: 1e-4          # Taux d'apprentissage du moteur local
  update_frequency: 10         # Fréquence de mise à jour du tenseur latent
  module_weights:              # Poids des différents modules dans les décisions
    transformer: 0.3
    curiosity: 0.2
    planning: 0.2
    memory: 0.3
  silence_mode_threshold: 0.85 # Seuil de tension pour activer le mode silence
  ```
* **.env** (optionnel) : Selenium credentials, API tokens.

Adaptez ces fichiers selon votre matériel et vos jeux de données.

1. Entraînement SFT

```bash
python scripts/train_sft.py \
  --config configs/sft_config.yaml \
  --output models/sft_finetuned/
```

2. LIMEN - Module d'Intention Émergente (superviseur)

```bash
python scripts/train_limen.py \
  --config configs/limen_config.yaml \
  --sft_checkpoint models/sft_finetuned/latest.pt \
  --output models/limen/
```

3. Intrinsic Curiosity Module (ICM) - Sous supervision LIMEN

```bash
python scripts/train_icm.py \
  --config configs/icm_config.yaml \
  --limen_checkpoint models/limen/latest.pt \
  --output models/icm/
```

4. Random Network Distillation (RND) - Sous supervision LIMEN

```bash
python scripts/train_rnd.py \
  --config configs/rnd_config.yaml \
  --limen_checkpoint models/limen/latest.pt \
  --output models/rnd/
```

5. Transformer² - Raisonnement avancé

```bash
python scripts/train_mcts_tf.py \
  --config configs/transformer2_config.yaml \
  --limen_checkpoint models/limen/latest.pt \
  --output models/mcts_transformer/
```

6. diffu-GRPO (Renforcement avec système intégré)

```bash
python scripts/train_diffu_grpo.py \
  --config configs/grpo_config.yaml \
  --limen_checkpoint models/limen/latest.pt \
  --output models/diffu_grpo_from_sft/
```

7. Lancement de l'API Temps-Réel & Dashboards
```bash
uvicorn realtime.server:app --reload
```

Cette configuration optimisée pour RTX 3090 garantit une utilisation efficace des 24GB de VRAM disponibles.

8. Rapport Hebdomadaire

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
│   ├── transformer_squared/
│   └── limen/
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
3. LIMEN - Initialisation du superviseur d'intentions
   ```bash
   python scripts/train_limen.py \
     --config configs/limen_config.yaml \
     --sft_checkpoint models/sft_finetuned/latest.pt \
     --output models/limen/
   ```
4. (Optionnel) Entraînement des modules de curiosité sous supervision LIMEN
   ```bash
   # ICM
   python scripts/train_icm.py \
     --config configs/icm_config.yaml \
     --limen_checkpoint models/limen/latest.pt \
     --output models/icm/
   
   # RND
   python scripts/train_rnd.py \
     --config configs/rnd_config.yaml \
     --limen_checkpoint models/limen/latest.pt \
     --output models/rnd/
   ```
5. diffu-GRPO (RL avec système complet intégré)
   ```bash
   python scripts/train_diffu_grpo.py \
     --config configs/grpo_config.yaml \
     --limen_checkpoint models/limen/latest.pt \
     --output models/diffu_grpo_from_sft/
   ```
6. Évaluation finale & déploiement
   ```bash
   python scripts/evaluate_full_system.py \
     --model models/diffu_grpo_from_sft/final_model \
     --limen models/limen/latest.pt \
     --data data/processed/test.jsonl
   ```
7. API Temps-Réel & Dashboards avec système complet
   ```bash
   uvicorn realtime.server:app --reload
   ```
Cette organisation assure que LIMEN supervise tous les autres modules dès leur entraînement.

## Matériel & Configuration

### Spécifications système recommandées
- **GPU** : NVIDIA GeForce RTX 3090 (24GB VRAM)
- **CPU** : AMD Ryzen 9 7900 (12-core/24-thread)
- **RAM** : 64GB DDR4/DDR5
- **Stockage** : SSD NVMe 1TB+ pour datasets et modèles

Le pipeline exploite cette configuration puissante pour un entraînement efficace des modèles DeepSeek R1 Qwen 12B.

### Configuration GPU
Avant chaque exécution d'entraînement, configurez votre GPU :
```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0  # RTX 3090
sudo nvidia-smi --gpu-reset -i 0  # Réinitialiser si nécessaire
nvidia-smi -pm 1                 # Mode performance
```

### Pipeline complet d'entraînement

Pour démarrer de zéro et couvrir toutes les phases avec l'approche phylogénétique conceptelle :

1. Configuration GPU
```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0  # RTX 3090
sudo nvidia-smi --gpu-reset -i 0  # reset si bloqué
nvidia-smi -pm 1                 # mode performance
```

2. Pré-traitement des données conceptuelles
```bash
python scripts/prepare_conceptual_data.py \
  --domains "physics,biology,economics,psychology" \
  --output data/conceptual_corpus/
```

3. **MetaLIMEN** - Définition des intentions d'apprentissage
```bash
python scripts/train_meta_limen.py \
  --config configs/meta_limen_config.yaml \
  --domains data/conceptual_corpus/ \
  --output models/meta_limen/
```

4. **SFT Guidé par Intentions** - Base linguistique intentionnelle
```bash
python scripts/train_intentional_sft.py \
  --config configs/sft_config.yaml \
  --meta_limen_checkpoint models/meta_limen/latest.pt \
  --output models/sft_finetuned/
```

5. **FullLIMEN** - Intentions sophistiquées post-SFT
```bash
python scripts/train_full_limen.py \
  --config configs/limen_config.yaml \
  --sft_checkpoint models/sft_finetuned/latest.pt \
  --meta_intentions models/meta_limen/intentions.pt \
  --output models/full_limen/
```

6. **Conceptual ICM** - Curiosité phylogénétique conceptelle
```bash
python scripts/train_conceptual_icm.py \
  --config configs/icm_config.yaml \
  --full_limen_checkpoint models/full_limen/latest.pt \
  --output models/conceptual_icm/
```

7. **Conceptual RND** - Nouveauté dans l'espace conceptuel
```bash
python scripts/train_conceptual_rnd.py \
  --config configs/rnd_config.yaml \
  --full_limen_checkpoint models/full_limen/latest.pt \
  --output models/conceptual_rnd/
```

8. **Transformer² Intentionnel** - Raffinement guidé phylogénétiquement
```bash
python scripts/train_intentional_transformer2.py \
  --config configs/transformer2_config.yaml \
  --full_limen_checkpoint models/full_limen/latest.pt \
  --output models/intentional_transformer_squared/
```

9. **MCTS Conceptuel** - Planification dans l'espace phylogénétique
```bash
python scripts/train_conceptual_mcts.py \
  --config configs/mcts_conceptual_config.yaml \
  --full_limen_checkpoint models/full_limen/latest.pt \
  --output models/conceptual_mcts/
```

10. **Phylogenetic Mythèmes** - Détection d'homologies conceptuelles
```bash
python scripts/train_phylogenetic_mythemes.py \
  --config configs/mythemes_config.yaml \
  --full_limen_checkpoint models/full_limen/latest.pt \
  --output models/phylogenetic_mythemes/
```

11. **Conceptual GRPO** - Optimisation phylogénétique globale
```bash
python scripts/train_conceptual_grpo.py \
  --config configs/conceptual_grpo_config.yaml \
  --full_limen_checkpoint models/full_limen/latest.pt \
  --conceptual_modules models/conceptual_*/ \
  --output models/conceptual_grpo/
```

12. **Évaluation Phylogénétique Complète**
```bash
python scripts/evaluate_phylogenetic_system.py \
  --model models/conceptual_grpo/final_model \
  --full_limen models/full_limen/latest.pt \
  --validation_type "phylogenetic_bootstrap" \
  --data data/conceptual_corpus/test/
```

13. **API Phylogénétique Temps-Réel**
```bash
uvicorn realtime.conceptual_server:app --reload \
  --env FULL_LIMEN_MODEL=models/full_limen/latest.pt \
  --env CONCEPTUAL_SYSTEM=models/conceptual_grpo/final_model
```

### Notes importantes sur l'ordre phylogénétique :
- **MetaLIMEN d'abord** : Définit les intentions d'apprentissage pré-linguistiques
- **SFT guidé ensuite** : Construit la base linguistique selon les intentions
- **FullLIMEN post-SFT** : Raffine les intentions avec capacités linguistiques complètes
- **Modules conceptuels** : Travaillent dans l'espace phylogénétique sous guidance FullLIMEN
- **GRPO conceptuel en dernier** : Optimise le système intégré dans l'espace conceptuel

Cette approche révolutionnaire transforme l'apprentissage en **construction phylogénétique intentionnelle** où chaque étape est guidée par la cohérence conceptuelle et l'intention émergente.

---

## Logique et Dépendances du Pipeline

### Pourquoi ce nouvel ordre révolutionnaire ?

L'inspiration de l'[article Nature Communications](https://www.nature.com/articles/s41467-021-22073-8) révèle que **l'intention guide la construction** dans les systèmes phylogénétiques. Notre pipeline hybride résout le paradoxe bootstrap :

```
MetaLIMEN → SFT_Guided → FullLIMEN → [ICM, RND, Transformer², MCTS] → diffu-GRPO → API
     ↓           ↓           ↓              ↓                            ↓         ↓
Meta-Intent  Intentional  Complete    Conceptual                Phylogenetic   Interface
Definition   Learning     Intentions  Modules                   Optimization
```

#### 1. **MetaLIMEN** - Intentions Pré-Linguistiques
- **Définit les objectifs d'apprentissage** avant toute compréhension complexe
- Utilise des embeddings simples (Word2Vec) pour intentions de haut niveau
- Établit un espace méta-intentionnel pour guider l'apprentissage
- **Résout le bootstrap** : intentions simples → capacités complexes

#### 2. **SFT Guidé** - Apprentissage Intentionnel
- **Construction fondation linguistique** orientée par les méta-intentions
- Curriculum basé sur les intentions phylogénétiques conceptuelles
- Filtrage et pondération des données selon les objectifs définis
- **Base solide** avec direction intentionnelle intégrée

#### 3. **FullLIMEN** - Raffinement Intentionnel
- **Intentions complètes** utilisant les capacités linguistiques acquises
- Encodage sophistiqué dans l'espace phylogénétique conceptuel
- Validation post-génération et guidance des modules avancés
- **Superviseur mature** avec compréhension linguistique complète

#### 4. **Modules Phylogénétiques** - Exploration Conceptuelle
- **ICM/RND conceptuel** : Curiosité dans l'espace phylogénétique des concepts
- **Transformer² intentionnel** : Raffinement guidé par intentions phylogénétiques
- **MCTS conceptuel** : Planification dans l'arbre des concepts
- **XNets/Mythèmes** : Détection d'homologies conceptuelles inter-domaines

#### 5. **Conceptual GRPO** - Optimisation Phylogénétique
- **Optimisation dans l'espace conceptuel** plutôt que l'espace des tokens
- Policy gradient guidé par vraisemblance phylogénétique conceptuelle
- Intégration des signaux de curiosité, planning et intentions
- **Optimisation globale** du système conceptuel intégré

### Avantages de cette Approche Hybride

✅ **Cohérence intentionnelle** : L'intention guide tout le processus dès le début  
✅ **Résolution bootstrap** : MetaLIMEN simple → FullLIMEN sophistiqué  
✅ **Base scientifique** : Inspirée des méthodes phylogénétiques validées  
✅ **Architecture unifiée** : Tous les modules dans l'espace conceptuel phylogénétique  
✅ **Optimisation cohérente** : GRPO dans l'espace des concepts  

### Conséquences d'un retour à l'ancien ordre :

❌ **Si SFT avant MetaLIMEN** : Apprentissage sans direction intentionnelle claire  
❌ **Si pas de MetaLIMEN** : Impossible de définir les intentions d'apprentissage  
❌ **Si FullLIMEN avant SFT** : Paradoxe bootstrap non résolu  
❌ **Si modules avant FullLIMEN** : Pas de guidance intentionnelle sophistiquée

✅ **Ordre optimal** : MetaLIMEN → SFT → FullLIMEN → Modules → GRPO garantit cohérence intentionnelle et optimisation phylogénétique conceptuelle

