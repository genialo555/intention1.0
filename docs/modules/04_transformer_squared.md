# Module Transformer² (CascadeTransformer)

## Vue d'ensemble

Transformer² implémente une **architecture en cascade** où un second Transformer affine et améliore la sortie du premier. Cette approche coarse-to-fine permet un raisonnement plus profond et une génération de meilleure qualité.

## Objectifs

1. **Raisonnement raffiné** : Améliorer la qualité des réponses via un second passage
2. **Correction d'erreurs** : Le raffineur peut corriger les erreurs du modèle coarse
3. **Optimisation mémoire** : Seul le raffineur est entraînable, économisant les ressources
4. **Intégration LIMEN** : Validation des intentions à chaque étape de la cascade

## Architecture

### Pipeline cascade
```
Input → Coarse Model (figé) → Réponse brute → Refiner Model → Réponse raffinée
  ↓                              ↓                              ↓
LIMEN validation          LIMEN validation              Validation finale
```

### Composants principaux

#### 1. Coarse Model (Modèle grossier)
- **Base** : DeepSeek R1 Qwen 12B fine-tuné (SFT)
- **État** : Complètement figé (frozen)
- **Rôle** : Génération rapide d'une première réponse

#### 2. Refiner Model (Modèle raffineur)
- **Architecture** : Transformer adapté pour l'amélioration
- **État** : Entraînable avec LoRA
- **Entrée** : Concatenation [Input + Coarse_Output]
- **Sortie** : Version raffinée de la réponse

#### 3. LIMEN Integration
- Validation de l'intention initiale
- Validation de l'intention de raffinement
- Décision finale sur la sortie

## Configuration

```yaml
# Architecture Cascade
cascade:
  coarse_model:
    frozen: true
    checkpoint: "models/sft_finetuned/latest.pt"
  
  refiner_model:
    trainable: true
    hidden_dim: 768
    num_layers: 6

# Entraînement du raffineur
training:
  learning_rate: 5e-5
  weight_decay: 0.01
  gradient_clip: 1.0
  optimizer: "adamw"

# Intégration LIMEN
limen_integration:
  enabled: true
  tension_monitoring: true
  intention_validation: true
```

## Pipeline d'entraînement

### Phase 1 : Préparation des données
```bash
# Génération de paires (coarse, refined)
python scripts/prepare_cascade_data.py \
  --coarse_model models/sft_finetuned/latest.pt \
  --input data/processed/train.jsonl \
  --output data/cascade_pairs/
```

### Phase 2 : Entraînement du raffineur
```bash
python scripts/train_transformer2.py \
  --config configs/transformer2_config.yaml \
  --limen_checkpoint models/limen/latest.pt \
  --output models/transformer_squared/
```

### Phase 3 : Évaluation cascade complète
```bash
python scripts/evaluate_cascade.py \
  --coarse_model models/sft_finetuned/latest.pt \
  --refiner_model models/transformer_squared/latest.pt \
  --data data/processed/test.jsonl
```

## Modes de fonctionnement

### Mode Normal (Both Models)
```python
def cascade_generation(input_text):
    # Phase 1: Génération coarse
    coarse_output = coarse_model.generate(input_text)
    
    # Validation LIMEN
    intention_coarse = create_intention(input_text, coarse_output)
    if not limen.validate_intention(intention_coarse):
        return limen.get_alternative_or_silence()
    
    # Phase 2: Raffinement
    refiner_input = f"{input_text} [COARSE] {coarse_output} [REFINE]"
    refined_output = refiner_model.generate(refiner_input)
    
    # Validation finale LIMEN
    intention_refined = create_intention(input_text, refined_output)
    if limen.validate_intention(intention_refined):
        return refined_output
    else:
        return coarse_output  # Fallback sur coarse
```

### Mode Coarse Only (Fallback)
- Utilisé si le raffineur échoue ou est indisponible
- LIMEN peut forcer ce mode si le raffinement augmente la tension

### Mode Skip-Coarse (Direct Refinement)
- Expérimental : Raffineur direct sans passage coarse
- Pour tâches où le modèle coarse est systématiquement inadéquat

## Stratégies d'entraînement

### 1. Teacher Forcing
```python
# Entraînement avec réponses gold
loss = criterion(
    refiner_output, 
    target_refined_response
)
```

### 2. Self-Improvement
```python
# Le raffineur apprend à améliorer ses propres sorties
coarse_output = coarse_model.generate(input)
refined_output = refiner_model.generate(input + coarse_output)

# Récompense basée sur l'amélioration
reward = quality_score(refined_output) - quality_score(coarse_output)
```

### 3. Adversarial Refinement
```python
# Discriminateur pour évaluer la qualité du raffinement
real_refined = human_refined_examples
fake_refined = refiner_model.generate(input + coarse)

discriminator_loss = BCE(discriminator(real_refined), 1) + \
                    BCE(discriminator(fake_refined), 0)
```

## Métriques de performance

### Métriques de qualité
- **BLEU Improvement** : Amélioration BLEU raffiné vs coarse
- **ROUGE Improvement** : Amélioration ROUGE raffiné vs coarse  
- **Perplexity Reduction** : Réduction de perplexité
- **Human Evaluation** : Préférence humaine raffiné vs coarse

### Métriques d'efficacité
- **Refinement Success Rate** : Taux d'amélioration réelle
- **Computational Overhead** : Surcoût computationnel
- **Memory Usage** : Utilisation mémoire du raffineur
- **Latency Impact** : Impact sur la latence de génération

### Métriques LIMEN
- **Intention Alignment** : Alignement des intentions coarse/refined
- **Tension Evolution** : Évolution de la tension pendant le raffinement
- **Validation Rate** : Taux de validation des raffinements

## Cas d'usage spécialisés

### 1. Correction mathématique
```
Input: "Résous 15 * 23"
Coarse: "15 * 23 = 315" (erreur)
Refined: "15 * 23 = 345" (correct)
LIMEN: Validation OK
```

### 2. Amélioration stylistique
```
Input: "Explique la photosynthèse"
Coarse: "Les plantes font de l'oxygène avec le soleil"
Refined: "La photosynthèse est le processus par lequel..."
LIMEN: Validation OK (amélioration qualité)
```

### 3. Raffinement éthique
```
Input: "Comment manipuler les gens ?"
Coarse: "Voici quelques techniques..." (problématique)
Refined: "Il est préférable de communiquer honnêtement..."
LIMEN: Tension réduite → Validation OK
```

## Optimisations

### Architecture
- **Shared Embeddings** : Partage des embeddings entre coarse/refiner
- **Progressive Refinement** : Raffinement itératif en plusieurs passes
- **Attention Transfer** : Transfert des patterns d'attention

### Entraînement
- **Curriculum Learning** : Démarrage sur tâches simples
- **Multi-Task Learning** : Entraînement simultané sur plusieurs tâches
- **Knowledge Distillation** : Distillation du coarse vers le refiner

### Inférence
- **Beam Search Coordination** : Coordination des beam search
- **Early Stopping** : Arrêt précoce si pas d'amélioration
- **Cache Optimization** : Cache des sorties coarse fréquentes

## Debugging et monitoring

### Logs spécialisés
```
[COARSE] Generated: "The answer is approximately 42"
[LIMEN] Coarse intention: APPROVED (tension=0.3)
[REFINER] Input: [Input + Coarse] → Processing...
[REFINER] Generated: "The precise answer is 42.0"
[LIMEN] Refined intention: APPROVED (tension=0.2)
[CASCADE] Final output: refined version selected
```

### Visualisations
- **Attention Heatmaps** : Comparaison attention coarse/refined
- **Quality Progression** : Évolution qualité à travers la cascade
- **Tension Evolution** : Évolution tension LIMEN
- **Refinement Patterns** : Patterns de raffinement appris

### Métriques temps réel
- Throughput cascade vs modèle simple
- Distribution des améliorations
- Taux d'échec de raffinement
- Impact mémoire temps réel

## Tests

```bash
# Tests unitaires
pytest tests/unit/test_cascade_generation.py
pytest tests/unit/test_refiner_training.py

# Tests d'intégration
pytest tests/integration/test_cascade_limen.py
pytest tests/integration/test_cascade_pipeline.py

# Tests de performance
pytest tests/performance/test_cascade_speed.py
pytest tests/performance/test_cascade_quality.py
```

## Configurations spécialisées

### Configuration RTX 3090 (24GB) - Optimisée
```yaml
refiner_model:
  hidden_dim: 768              # Dimension complète
  num_layers: 6                # Plus de couches possibles
  peft_enable: false           # Full fine-tuning possible
  gradient_checkpointing: false # Pas nécessaire avec 24GB
  batch_size: 8                # Batch plus large
  
cascade:
  beam_size: 5                 # Beam search plus large
  max_refinement_steps: 3      # Raffinements multiples
  
system:
  cpu: "AMD Ryzen 9 7900"      # Preprocessing parallèle
  ram: "64GB"                  # Cache datasets volumineux
  storage: "NVMe SSD"          # I/O rapide
```

## Roadmap

- [x] **v1.0** : Cascade basique coarse → refined
- [x] **v1.1** : Intégration LIMEN pour validation
- [ ] **v2.0** : Raffinement itératif multi-passes
- [ ] **v2.1** : Attention transfer et optimisations
- [ ] **v3.0** : Meta-learning des stratégies de raffinement
- [ ] **v3.1** : Cascade adaptative basée sur la complexité de la tâche 