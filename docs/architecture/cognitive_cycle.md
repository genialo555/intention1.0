# Cycle Cognitif en 3 Phases

Ce document décrit comment chaque module du pipeline s'intègre dans un cycle de génération en trois phases : exploration (Curiosity), filtrage/pondération (Affect Modeling), stabilisation (Stabilization).

---

## 1. CuriosityFirstAgent (Exploration brute)

Modules inclus :
- **StochasticSamplingAgent** : échantillonnage aléatoire du modèle de base (Transformer).
- **Intrinsic Curiosity Module (ICM)** : générateur de récompenses intrinsèques pour encourager la nouveauté.
- **Random Network Distillation (RND)** : mesure la nouveauté par erreur de prédiction.
- **Transformer² (Cascade Transformer)** : pipeline coarse-to-fine proposant des raffinements initiaux.

### Objectifs :
- Explorer l'espace des réponses sans filtre.
- Favoriser la diversité et la découverte de nouvelles pistes.

---

## 2. AffectModelingAgent (Filtrage et Pondération)

Modules inclus :
- **FullLIMEN** : affinage des intentions avec capacités linguistiques complètes.
- **DomainIntentionMapper & SimpleCurriculumGenerator** : calcul des priorités et pondération des intentions.
- **SFT Guided** : fine-tuning guidé par MetaLIMEN.

### Objectifs :
- Pondérer les réponses candidates selon leur alignement avec l'intention courante.
- Filtrer les séquences peu pertinentes ou incohérentes.

---

## 3. StabilizationAgent (Consolidation et Planification)

Modules inclus :
- **Phylogenetic Group Think** : collaboration token-level d'agents (ICM, RND, T², MCTS, XNets).
- **MCTS & Tree-of-Thoughts** : planification multi-étapes cohérente.
- **diffu-GRPO** : optimisation par renforcement dans l'espace conceptuel.
- **Spaced Repetition & Continual Learning** : ancrage mémoire à long terme.

### Objectifs :
- Consolider une réponse stable et cohérente.
- Appliquer les modules de planification et d'optimisation conceptuelle.

---

## Orchestration & Méta-Contrôle

Le Méta-Contrôleur (`orchestration/controller.py`) gère le cycle :
1. **CuriosityFirstAgent** → génère `candidate_responses` via exploration aléatoire et curiosité.
2. **AffectModelingAgent** → calcule `weighted_responses` en filtrant et pondérant selon les intentions.
3. **StabilizationAgent** → sélectionne la `final_response` optimale et cohérente.
4. Réinitialisation du cycle en cas de « reset intent » déclenché par un événement externe.

### Problématiques clés :
- **Exploration vs exploitation** : calibrer le nombre de candidats vs stabilité.
- **Signaux affectifs** : définir les métriques (trust, energy, cohérence, feedback utilisateur).
- **Latence** : maintenir un temps d'inférence acceptable malgré la complexité.
- **Oscillations d'intention** : éviter des basculements répétitifs sans convergence.

--- 