# Module SFT (Supervised Fine-Tuning)

## Vue d'ensemble

Le module SFT constitue la **fondation** du système Curiosity-Driven AI. Il fournit les capacités linguistiques de base nécessaires à toute communication cohérente et sert de point de départ pour tous les autres modules.

## Objectifs

1. **Base linguistique** : Établir les capacités de compréhension et génération de texte
2. **Raisonnement initial** : Développer l'intuition de raisonnement sur des tâches structurées
3. **Socle stable** : Fournir un checkpoint fiable pour l'initialisation des autres modules

## Architecture

### Modèle de base
- **DeepSeek R1 Qwen 12B** comme modèle fondamental
- Configuration PEFT/LoRA pour l'efficacité mémoire
- Support RTX 3090 (24GB VRAM) optimisé

### Datasets d'entraînement
- **GSM8K** : Problèmes mathématiques élémentaires
- **MATH** : Problèmes mathématiques avancés
- **Livres pour enfants** : Acquisition naturelle du langage
- **Énigmes logiques** : Développement du raisonnement

## Configuration

```yaml
model_name: deepseekr1-qwen-12b
batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 3e-5
epochs: 3
max_length: 512
peft_enable: true
lora_r: 16
```

## Pipeline d'entraînement

### 1. Préparation des données
```bash
python scripts/prepare_data.py \
  --input data/raw/ \
  --output data/sft_examples/
```

### 2. Entraînement
```bash
python scripts/train_sft.py \
  --config configs/sft_config.yaml \
  --output models/sft_finetuned/
```

### 3. Évaluation
```bash
python scripts/evaluate_sft.py \
  --model models/sft_finetuned/latest.pt \
  --data data/processed/val.jsonl
```

## Métriques de performance

- **Perplexité** : < 15 sur validation
- **BLEU Score** : > 0.7 pour génération
- **Accuracy** : > 85% sur GSM8K
- **Loss convergence** : Stabilisation après epoch 2

## Intégration système

### Sorties
- `models/sft_finetuned/latest.pt` : Checkpoint principal
- `models/sft_finetuned/tokenizer/` : Tokenizer adapté
- Métriques dans TensorBoard

### Dépendances suivantes
- **LIMEN** : Utilise le checkpoint SFT comme base
- **Modules spécialisés** : Tous initialisés à partir de SFT

## Optimisations GPU

### RTX 3090 (24GB VRAM) - Configuration optimisée
```yaml
batch_size: 8
gradient_accumulation_steps: 2
fp16: true
gradient_checkpointing: false  # Pas nécessaire avec 24GB
peft_enable: false              # Full fine-tuning possible
hidden_dim: 768                 # Dimension complète
num_layers: 12                  # Architecture complète
```

### Configuration système
- **CPU** : AMD Ryzen 9 7900 (excellent pour preprocessing)
- **RAM** : 64GB (gestion datasets volumineux)
- **GPU** : RTX 3090 24GB VRAM
- **I/O** : SSD NVMe pour accès rapide aux datasets

## Debugging et monitoring

### Logs importants
- Loss d'entraînement et validation
- Gradient norms
- Learning rate schedule
- Memory usage

### Problèmes courants
1. **OOM (Out of Memory)** : Réduire batch_size
2. **Loss plateau** : Ajuster learning_rate
3. **Instabilité** : Activer gradient_checkpointing

## Checkpoint et reprise

```bash
# Reprise depuis un checkpoint
python scripts/train_sft.py \
  --resume_from_checkpoint models/sft_finetuned/checkpoint-1000
```

## Tests

```bash
# Tests unitaires
pytest tests/unit/test_sft.py

# Tests d'intégration
pytest tests/integration/test_sft_pipeline.py
```

## Roadmap

- [x] **v1.0** : Entraînement de base sur GSM8K
- [x] **v1.1** : Support multi-GPU
- [ ] **v2.0** : Curriculum learning progressif
- [ ] **v2.1** : Intégration données web dynamiques 