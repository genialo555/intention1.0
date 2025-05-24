# Module diffu-GRPO (Diffusion Generalized Relative Policy Optimization)

## Vue d'ensemble

diffu-GRPO implémente l'**apprentissage par renforcement** pour l'optimisation globale du système Curiosity-Driven AI. Il s'agit d'une adaptation innovante du policy gradient aux modèles de type diffusion masquée, permettant l'optimisation fine du système complet intégré.

## Objectifs

1. **Optimisation globale** : Améliorer les performances du système complet via RL
2. **Reward shaping** : Intégrer les signaux de récompense multiples (LIMEN, curiosité, qualité)
3. **Stabilité d'entraînement** : Éviter le mode collapse et maintenir la diversité
4. **Intégration LIMEN** : Utiliser les tensions comme signal de récompense

## Principe théorique

### Diffusion masquée adaptée au RL
```
Policy π_θ : (state, context) → action_distribution
State = [input_text, module_history, LIMEN_state]
Action = [masked_tokens, generation_strategy]
Reward = f(quality, LIMEN_tension, curiosity, human_feedback)
```

### Policy gradient adapté
```
∇J(θ) = E[∇log π_θ(a|s) * (R - b(s))]

Où :
- R = reward total (weighted sum)
- b(s) = baseline (value function)
- π_θ = policy paramétrée
```

## Architecture

### Composants principaux

#### 1. Policy Network
- **Base** : DeepSeek R1 Qwen 12B fine-tuné
- **Adaptation** : Couches LoRA pour policy gradient
- **Sortie** : Distribution sur actions (masking + génération)

#### 2. Value Network
- **Rôle** : Estimation de la valeur des états
- **Architecture** : Réseau critique léger
- **Entraînement** : Regression vers rewards observées

#### 3. Reward Model
- **Composite** : Agrégation de multiples signaux
- **Components** :
  - Quality score (BLEU, ROUGE, human eval)
  - LIMEN tension (negative reward pour tensions élevées)
  - Curiosity bonus (ICM/RND signals)
  - Safety score (content filtering)

#### 4. Masking Strategy
- **Dynamic masking** : Probabilité de masquage adaptative
- **Informed masking** : Masquage basé sur l'attention
- **Progressive masking** : Curriculum de difficulté

## Configuration

```yaml
# diffu-GRPO Configuration
model_path: models/sft_finetuned/latest.pt
limen_checkpoint: models/limen/latest.pt

# Policy gradient
learning_rate: 5e-5
beta1: 0.9
beta2: 0.999
gradient_clip: 1.0

# Masking strategy
mask_rate: 0.15              # Proportion tokens masqués
dynamic_masking: true        # Masquage adaptatif
informed_masking: true       # Basé sur attention

# Reward weighting
reward_weights:
  quality: 0.4               # Qualité génération
  limen_tension: -0.3        # Tension LIMEN (négatif)
  curiosity: 0.2             # Bonus curiosité
  safety: 0.1                # Score sécurité

# Training parameters
timesteps: 1000              # Steps d'entraînement
batch_size: 8                # Taille batch
ppo_epochs: 4                # Epochs PPO par batch
```

## Pipeline d'entraînement

### Phase 1 : Préparation reward model
```bash
# Collection données préférence humaine
python scripts/collect_human_preferences.py \
  --model models/sft_finetuned/latest.pt \
  --output data/preferences/

# Entraînement reward model
python scripts/train_reward_model.py \
  --preferences data/preferences/ \
  --output models/reward_model/
```

### Phase 2 : Entraînement diffu-GRPO
```bash
python scripts/train_diffu_grpo.py \
  --config configs/grpo_config.yaml \
  --limen_checkpoint models/limen/latest.pt \
  --reward_model models/reward_model/latest.pt \
  --output models/diffu_grpo/
```

### Phase 3 : Évaluation comparative
```bash
python scripts/evaluate_rl_impact.py \
  --baseline models/sft_finetuned/latest.pt \
  --rl_model models/diffu_grpo/latest.pt \
  --benchmarks data/eval_suite/
```

## Stratégies d'entraînement

### 1. Proximal Policy Optimization (PPO)
```python
def ppo_update(policy, value_fn, batch):
    advantages = compute_advantages(batch.rewards, batch.values)
    
    for epoch in range(ppo_epochs):
        # Policy loss avec clipping
        ratio = policy.prob(batch.actions) / batch.old_probs
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = F.mse_loss(value_fn(batch.states), batch.returns)
        
        # Update
        total_loss = policy_loss + value_coef * value_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
```

### 2. LIMEN-guided reward shaping
```python
def compute_reward(generation, limen_state, context):
    # Qualité de base
    quality_reward = compute_quality_score(generation)
    
    # Tension LIMEN (pénalité)
    tension_penalty = -limen_state.tension * tension_weight
    
    # Bonus curiosité
    curiosity_bonus = compute_curiosity_bonus(generation, context)
    
    # Score sécurité
    safety_score = compute_safety_score(generation)
    
    return (quality_reward + tension_penalty + 
            curiosity_bonus + safety_score)
```

### 3. Curriculum learning
```python
def get_masking_schedule(step, total_steps):
    # Démarrage avec masquage facile
    base_rate = 0.10
    max_rate = 0.25
    
    # Progression linéaire
    progress = step / total_steps
    return base_rate + (max_rate - base_rate) * progress
```

## Métriques de performance

### Métriques RL
- **Average Reward** : Récompense moyenne par épisode
- **Policy Entropy** : Entropie de la politique (diversité)
- **Value Accuracy** : Précision du value network
- **KL Divergence** : Divergence vs politique initiale

### Métriques qualité
- **Generation Quality** : BLEU, ROUGE, perplexité
- **Human Preference** : Win rate vs baseline
- **LIMEN Coherence** : Tension moyenne post-RL
- **Safety Score** : Taux de contenu problématique

### Métriques système
- **Convergence Speed** : Steps pour convergence
- **Training Stability** : Variance des rewards
- **Memory Efficiency** : Utilisation mémoire GPU
- **Throughput** : Samples/seconde d'entraînement

## Challenges et solutions

### 1. Mode Collapse
**Problème** : La politique converge vers une solution sous-optimale

**Solutions** :
- Entropy regularization
- Diverse beam search rewards
- Multiple random seeds
- Early stopping basé sur diversité

### 2. Reward Hacking
**Problème** : Exploitation de failles dans le reward model

**Solutions** :
- Multiple reward models (ensemble)
- Human-in-the-loop validation
- Adversarial testing
- LIMEN supervision des rewards

### 3. Training Instability
**Problème** : Instabilité pendant l'entraînement RL

**Solutions** :
- Gradient clipping adaptatif
- Learning rate scheduling
- Batch normalization
- Checkpoint averaging

## Intégration LIMEN avancée

### Reward shaping intelligent
```python
class LIMENGuidedReward:
    def __init__(self, limen_model):
        self.limen = limen_model
        
    def compute_reward(self, state, action, next_state):
        # Récompense de base
        base_reward = self.base_reward_fn(state, action, next_state)
        
        # Validation LIMEN
        intention = create_intention(state, action)
        limen_result = self.limen.validate_intention(intention)
        
        if limen_result.mode == "APPROVED":
            return base_reward
        elif limen_result.mode == "ALTERNATIVE":
            return base_reward * 0.5  # Pénalité modérée
        else:  # SILENCE
            return -1.0  # Forte pénalité
```

### Meta-learning des rewards
```python
def meta_update_rewards(limen_feedback, episode_outcomes):
    """Ajuste les poids de reward basé sur feedback LIMEN"""
    for episode in episode_outcomes:
        if episode.limen_tension > threshold:
            # Réduire le poids des rewards qui ont causé tension
            adjust_reward_weights(episode.actions, -0.1)
        else:
            # Renforcer les rewards des actions approuvées
            adjust_reward_weights(episode.actions, +0.05)
```

## Evaluation et benchmarking

### Benchmarks standards
- **MATH** : Problèmes mathématiques
- **GSM8K** : Problèmes arithmétiques
- **HumanEval** : Génération de code
- **TruthfulQA** : Véracité des réponses

### Métriques spécialisées
- **Curiosity-driven exploration** : Coverage de l'espace conceptuel
- **LIMEN coherence** : Stabilité des validations
- **Transfer learning** : Performance cross-domain
- **Safety evaluation** : Résistance aux prompts adverses

### Études utilisateurs
- **Preference studies** : Comparaisons humaines
- **Engagement metrics** : Durée d'interaction
- **Trust calibration** : Confiance utilisateur
- **Error analysis** : Types d'erreurs post-RL

## Optimisations avancées

### 1. Distributed training
```python
# Multi-GPU policy gradient
class DistributedPPO:
    def __init__(self, world_size):
        self.world_size = world_size
        
    def collect_experiences(self):
        # Collecte parallèle sur multiple GPUs
        experiences = gather_from_all_ranks(local_experiences)
        return experiences
        
    def update_policy(self, experiences):
        # Gradient averaging across GPUs
        gradients = compute_gradients(experiences)
        avg_gradients = all_reduce(gradients) / self.world_size
        apply_gradients(avg_gradients)
```

### 2. Memory-efficient training
- Gradient checkpointing pour policy network
- Mixed precision (fp16) training
- Gradient accumulation pour large batch sizes
- Experience replay avec compression

### 3. Adaptive hyperparameters
```python
def adapt_hyperparameters(training_stats):
    # Ajustement automatique basé sur performance
    if training_stats.kl_divergence > threshold:
        reduce_learning_rate()
    if training_stats.entropy < min_entropy:
        increase_entropy_coefficient()
```

## Tests et validation

```bash
# Tests unitaires RL
pytest tests/unit/test_policy_gradient.py
pytest tests/unit/test_reward_computation.py

# Tests d'intégration
pytest tests/integration/test_diffu_grpo_limen.py
pytest tests/integration/test_rl_pipeline.py

# Tests de performance
pytest tests/performance/test_rl_speed.py
pytest tests/performance/test_memory_usage.py

# Tests de stabilité
pytest tests/stability/test_convergence.py
pytest tests/stability/test_mode_collapse.py
```

## Debugging et monitoring

### Logs RL spécialisés
```
[RL] Episode 1247: avg_reward=0.73, entropy=2.1, kl_div=0.02
[LIMEN] Tension evolution: 0.6 → 0.4 (improvement)
[REWARD] Components: quality=0.8, tension=-0.2, curiosity=0.1
[POLICY] Action distribution: mask_rate=0.18, strategy=beam_search
```

### Visualisations temps réel
- **Reward curves** : Évolution des récompenses
- **Policy entropy** : Diversité de la politique
- **LIMEN tension** : Évolution de la cohérence
- **Action distributions** : Patterns d'actions apprises

### Alertes automatiques
- Détection mode collapse
- Monitoring divergence KL
- Alertes stabilité training
- Performance regression detection

## Roadmap

- [x] **v1.0** : PPO de base avec reward composite
- [x] **v1.1** : Intégration LIMEN dans reward shaping
- [ ] **v2.0** : Meta-learning des weights de reward
- [ ] **v2.1** : Multi-objective optimization (Pareto fronts)
- [ ] **v3.0** : Hierarchical RL avec sous-politiques
- [ ] **v3.1** : Continual RL avec experience replay intelligent 