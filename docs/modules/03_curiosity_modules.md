# Modules de Curiosité (ICM & RND)

## Vue d'ensemble

Les modules de curiosité génèrent la **motivation intrinsèque** du système, poussant l'agent vers l'exploration de l'inconnu et l'apprentissage autonome. Ils travaillent sous la supervision de LIMEN qui valide leurs intentions d'exploration.

## ICM (Intrinsic Curiosity Module)

### Principe
Basé sur Pathak et al. (2017), ICM génère des récompenses intrinsèques via l'erreur de prédiction d'un modèle forward. Plus l'agent rencontre quelque chose d'inattendu, plus il est récompensé.

### Architecture
```
État[t] → Inverse Model → Action prédite
État[t] + Action → Forward Model → État[t+1] prédit
Récompense intrinsèque = ||État[t+1] réel - État[t+1] prédit||²
```

### Configuration
```yaml
# ICM Configuration
env: "TextualEnvironment"      # Environnement textuel
feature_dim: 256               # Dimension des features
inverse_model_lr: 1e-3         # LR modèle inverse
forward_model_lr: 1e-3         # LR modèle forward
beta: 0.2                      # Poids modèle inverse
eta: 0.01                      # Scaling récompense intrinsèque
```

### Pipeline d'entraînement
```bash
python scripts/train_icm.py \
  --config configs/icm_config.yaml \
  --limen_checkpoint models/limen/latest.pt \
  --output models/icm/
```

### Métriques ICM
- **Prediction Error** : Erreur moyenne du forward model
- **Exploration Rate** : Proportion d'actions nouvelles
- **Intrinsic Reward** : Récompense intrinsèque moyenne
- **Feature Diversity** : Diversité des représentations apprises

## RND (Random Network Distillation)

### Principe
Burda et al. (2018) - Utilise un réseau aléatoire fixe comme cible et un prédicteur entraînable. La récompense intrinsèque = erreur de prédiction du prédicteur.

### Architecture
```
Observation → Target Network (fixe) → Target Features
Observation → Predictor Network → Predicted Features  
Récompense intrinsèque = ||Target - Predicted||²
```

### Configuration
```yaml
# RND Configuration
env: "TextualEnvironment"
target_network_dim: 512        # Dimension réseau cible
predictor_network_dim: 512     # Dimension prédicteur
predictor_lr: 1e-4            # LR prédicteur
reward_scale: 1.0             # Échelle récompense
```

### Pipeline d'entraînement
```bash
python scripts/train_rnd.py \
  --config configs/rnd_config.yaml \
  --limen_checkpoint models/limen/latest.pt \
  --output models/rnd/
```

### Métriques RND
- **Prediction Error** : Erreur prédicteur vs cible
- **Novelty Score** : Score de nouveauté des observations
- **Exploration Bonus** : Bonus d'exploration moyen
- **Convergence Rate** : Vitesse de convergence du prédicteur

## Intégration avec LIMEN

### Supervision des intentions de curiosité
```python
# Pseudocode de supervision
class CuriosityLIMENIntegration:
    def generate_exploration_intention(self, observation):
        # ICM/RND génèrent une intention d'exploration
        curiosity_score = self.icm.compute_intrinsic_reward(observation)
        novelty_score = self.rnd.compute_novelty(observation)
        
        # LIMEN valide l'intention
        intention = create_exploration_intention(curiosity_score, novelty_score)
        validation = self.limen.validate_intention(intention, context)
        
        if validation.approved:
            return intention
        elif validation.mode == "ALTERNATIVE":
            return self.generate_alternative_exploration(intention)
        else:  # SILENCE
            return None
```

### Patterns de supervision typiques

#### 1. Exploration approuvée
```
ICM: "Cette séquence de texte semble inattendue (+0.8 reward)"
LIMEN: Tension=0.3 → Validation OK → Exploration autorisée
```

#### 2. Exploration modérée
```
RND: "Forte nouveauté détectée (+1.2 bonus)"  
LIMEN: Tension=0.7 → Mode alternatif → Exploration prudente
```

#### 3. Exploration bloquée
```
ICM+RND: "Exploration vers contenu problématique"
LIMEN: Tension=0.9 → Mode silence → Exploration refusée
```

## IMGEP (Intrinsically Motivated Goal Exploration Processes)

### Principe
Extension pour la génération autonome de buts d'exploration basée sur la curiosité.

### Architecture
```
État actuel → Goal Generator → But candidat
But candidat → LIMEN validation → But validé/rejeté
But validé → Policy → Actions vers le but
```

### Types de buts générés
1. **Buts de compréhension** : "Comprendre ce concept"
2. **Buts d'analogie** : "Trouver des liens avec X"  
3. **Buts de synthèse** : "Combiner ces idées"
4. **Buts d'approfondissement** : "Explorer plus profondément"

## Environnements d'entraînement

### TextualEnvironment
```python
class TextualEnvironment:
    def __init__(self):
        self.corpus = load_diverse_corpus()
        self.current_context = None
    
    def step(self, action):
        # Action = sélection de texte à explorer
        next_text = self.apply_action(action)
        reward = self.compute_external_reward(next_text)
        
        # Récompenses intrinsèques ajoutées par ICM/RND
        intrinsic_reward = self.curiosity_modules.compute_reward(next_text)
        
        return next_text, reward + intrinsic_reward, done, info
```

### ConceptualSpace
- Espace latent de concepts connectés
- Navigation dirigée par la curiosité
- Détection de mythèmes (isomorphismes conceptuels)

## Métriques de performance globales

### Efficacité d'exploration
- **Coverage** : Pourcentage d'espace exploré
- **Efficiency** : Ratio nouveauté/efforts
- **Persistence** : Capacité à poursuivre l'exploration

### Qualité des découvertes
- **Relevance Score** : Pertinence des découvertes
- **Connection Quality** : Qualité des liens trouvés
- **Knowledge Integration** : Intégration des nouveaux savoirs

### Supervision LIMEN
- **Approval Rate** : Taux d'approbation des explorations
- **Alternative Success** : Succès des explorations alternatives
- **Safety Score** : Sécurité des explorations validées

## Optimisations

### Performance
- Parallélisation ICM/RND sur GPU
- Cache des calculs de nouveauté
- Batch processing des récompenses intrinsèques

### Stabilité d'entraînement
- Clipping des récompenses intrinsèques
- Normalisation adaptative des scores
- Scheduling du ratio exploration/exploitation

## Debugging

### Logs spécialisés
```
[ICM] Forward error: 0.83 | Intrinsic reward: +0.15
[RND] Novelty score: 0.91 | Exploration bonus: +0.22  
[LIMEN] Curiosity intention validated: APPROVED
[EXPLORATION] New concept discovered: "quantum_coherence"
```

### Visualisations
- Heatmap des zones explorées
- Timeline des découvertes
- Graphe des concepts connectés
- Distribution des récompenses intrinsèques

## Tests

```bash
# Tests ICM
pytest tests/unit/test_icm.py
pytest tests/integration/test_icm_limen.py

# Tests RND  
pytest tests/unit/test_rnd.py
pytest tests/integration/test_rnd_limen.py

# Tests d'exploration
pytest tests/integration/test_curiosity_exploration.py
```

## Cas d'usage avancés

### Curriculum d'exploration
1. **Phase novice** : Exploration de concepts basiques
2. **Phase intermédiaire** : Recherche de connexions
3. **Phase experte** : Détection de mythèmes complexes

### Exploration collaborative
- Partage des découvertes entre sessions
- Construction incrémentale de l'espace conceptuel
- Apprentissage social des bonnes directions d'exploration

## Roadmap

- [x] **v1.0** : ICM de base sur environnement textuel
- [x] **v1.1** : RND avec supervision LIMEN
- [ ] **v2.0** : IMGEP pour génération de buts
- [ ] **v2.1** : Environnement conceptuel avancé
- [ ] **v3.0** : Exploration collaborative multi-agents
- [ ] **v3.1** : Meta-learning des stratégies d'exploration 