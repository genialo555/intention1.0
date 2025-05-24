# Module FullLIMEN (Latent Intent Meta-Emergent Nexus)

## Vue d'ensemble

FullLIMEN constitue l'**organe de cohérence intentionnelle sophistiquée** du système, raffinant les méta-intentions simples en intentions conceptuelles complexes grâce aux capacités linguistiques acquises lors du SFT. Il s'appuie sur l'[approche phylogénétique validée](https://www.nature.com/articles/s41467-021-22073-8) pour organiser l'espace conceptuel.

## Objectifs

1. **Raffinement intentionnel** : Transformer les méta-intentions simples en intentions conceptuelles sophistiquées
2. **Espace phylogénétique conceptuel** : Organiser les concepts dans un arbre phylogénétique pour navigation intentionnelle
3. **Validation post-génération** : Valider la cohérence des sorties générées avec les intentions
4. **Guidance conceptuelle** : Guider les modules avancés dans l'espace phylogénétique conceptuel

## Architecture fonctionnelle

### Position dans le Pipeline

```
MetaLIMEN → SFT_Guided → **FullLIMEN** → [ICM, RND, Transformer², MCTS] → Conceptual-GRPO
     ↓             ↓          ↓                ↓                            ↓
Meta-Intent   Linguistic   Sophisticated   Conceptual                Phylogenetic
Definition    Foundation   Intentions      Modules                   Optimization
```

### Composants principaux

| Composant | Fonction |
|-----------|----------|
| **Conceptual Encoder** | Encode intentions sophistiquées (dim=512, heads=12) |
| **Phylogenetic Space** | Espace conceptuel organisé phylogénétiquement |
| **Intention Refinement** | Raffinement méta-intentions → intentions complexes |
| **Post-Gen Validation** | Validation post-génération cohérence |
| **SPR-Guided Exploration** | Navigation dans l'arbre conceptuel |

### Briques architecturales phylogénétiques

#### 1. Conceptual Phylogenetic Space
Inspiré de l'[article Nature Communications](https://www.nature.com/articles/s41467-021-22073-8) :

```python
class ConceptualPhylogeneticSpace:
    """
    Espace phylogénétique pour concepts, analogue aux arbres évolutionnaires
    """
    def __init__(self):
        self.concept_trees = {}  # Arbres par domaine
        self.spr_predictor = ConceptualSPRPredictor()  # ML-guided search
        
    def build_domain_tree(self, domain_concepts):
        """Construction arbre phylogénétique conceptuel"""
        # Matrice distance conceptuelle
        distance_matrix = self.compute_conceptual_distances(domain_concepts)
        
        # Arbre initial (Neighbor-Joining adapté)
        initial_tree = self.conceptual_neighbor_joining(distance_matrix)
        
        # Optimisation SPR guidée par ML
        optimized_tree = self.ml_guided_spr_optimization(initial_tree)
        
        return optimized_tree
```

#### 2. ML-Guided Conceptual Search
Adaptation directe de l'approche Nature Communications aux concepts :

```python
class ConceptualSPRPredictor:
    """
    Prédicteur ML pour mouvements SPR conceptuels
    Basé sur Random Forest avec 19 features conceptuelles
    """
    def __init__(self):
        self.ml_predictor = RandomForestRegressor()
        self.conceptual_features = CONCEPTUAL_FEATURES_19
        
    def predict_best_conceptual_moves(self, concept_tree, candidate_moves):
        """
        Prédit les meilleurs mouvements conceptuels sans calcul de vraisemblance
        """
        features = []
        for move in candidate_moves:
            move_features = self.extract_conceptual_features(
                concept_tree, move.pruned_concept, move.graft_position
            )
            features.append(move_features)
        
        # Prédiction ML des scores
        scores = self.ml_predictor.predict(features)
        
        # Retourne top-k mouvements prometteurs
        return self.select_top_k_moves(candidate_moves, scores, k=5)
```

#### 3. 19 Features Conceptuelles
Adaptation des features phylogénétiques aux concepts :

```python
CONCEPTUAL_FEATURES_19 = [
    # Topologiques (inspirées phylogénie)
    'concept_depth',           # Profondeur dans l'arbre conceptuel
    'subtree_size',           # Taille sous-arbre conceptuel  
    'branch_length',          # Distance sémantique
    'sister_similarity',      # Similarité avec concept "sister"
    
    # Sémantiques
    'embedding_norm',         # Norme du vecteur conceptuel
    'domain_frequency',       # Fréquence dans le domaine
    'cross_domain_score',     # Score trans-domaines
    'abstraction_level',      # Niveau d'abstraction
    
    # Structurelles  
    'dependency_count',       # Nombre de dépendances conceptuelles
    'analogy_strength',       # Force des analogies
    'transfer_potential',     # Potentiel de transfert
    'cognitive_complexity',   # Complexité cognitive
    
    # Dynamiques
    'usage_frequency',        # Fréquence d'utilisation
    'discovery_recency',      # Récence de découverte
    'evolution_rate',         # Taux d'évolution du concept
    'stability_score',        # Stabilité temporelle
    
    # Meta-conceptuelles
    'metaphor_richness',      # Richesse métaphorique
    'pattern_generality',     # Généralité du pattern
    'emergence_score'         # Score d'émergence
]
```

## Configuration

```yaml
# === Architecture du tenseur latent ===
latent_dim: 128              # Dimension du tenseur d'intention latente
hidden_dim: 512              # Dimension augmentée avec RTX 3090
num_heads: 12                # Plus de têtes avec 24GB VRAM
num_layers: 6                # Plus de couches possibles

# === Espace phylogénétique conceptuel ===
phylogenetic_space:
  domains: ["physics", "biology", "economics", "psychology"]
  spr_features: 19           # Features phylogénétiques conceptuelles
  ml_predictor: "random_forest"
  bootstrap_samples: 100     # Validation bootstrap

# === Raffinement intentionnel ===
intention_refinement:
  meta_to_full_mapping: true
  linguistic_enhancement: true
  conceptual_positioning: true
  phylogenetic_validation: true

# === Validation post-génération ===
post_generation:
  coherence_threshold: 0.8
  phylogenetic_likelihood: true
  intention_alignment: true
  safety_filtering: true
```

## Pipeline d'entraînement

### 1. Initialisation avec méta-intentions
```bash
python scripts/train_full_limen.py \
  --config configs/limen_config.yaml \
  --sft_checkpoint models/sft_finetuned/latest.pt \
  --meta_intentions models/meta_limen/intentions.pt \
  --output models/full_limen/
```

### 2. Phases d'entraînement phylogénétiques

#### Phase 1 : Construction espace conceptuel phylogénétique
- Construction arbres conceptuels par domaine
- Entraînement prédicteur SPR conceptuel
- Validation bootstrap des arbres conceptuels

#### Phase 2 : Raffinement intentions sophistiquées
- Mapping méta-intentions → intentions complexes
- Enhancement via capacités linguistiques SFT
- Positionnement dans l'espace phylogénétique

#### Phase 3 : Validation et optimisation
- Validation post-génération
- Optimisation seuils phylogénétiques
- Calibration guidance des modules avancés

## Modes de fonctionnement

### Mode Raffinement Intentionnel
```python
def refine_meta_intentions(self, meta_intentions, sft_capabilities):
    """
    Raffine les méta-intentions simples en intentions sophistiquées
    """
    refined_intentions = []
    
    for meta_intention in meta_intentions:
        # Enhancement linguistique via SFT
        linguistic_enhancement = self.enhance_with_sft(
            meta_intention, sft_capabilities
        )
        
        # Positionnement phylogénétique conceptuel
        phylo_position = self.position_in_conceptual_tree(
            linguistic_enhancement
        )
        
        # Validation bootstrap
        confidence = self.bootstrap_validate_intention(phylo_position)
        
        if confidence > self.confidence_threshold:
            refined_intentions.append({
                'intention': linguistic_enhancement,
                'phylo_position': phylo_position,
                'confidence': confidence
            })
    
    return refined_intentions
```

### Mode Guidance Conceptuelle
```python
def guide_conceptual_module(self, module_type, input_context):
    """
    Guide les modules avancés dans l'espace phylogénétique conceptuel
    """
    # Identification intention conceptuelle pertinente
    relevant_intentions = self.identify_relevant_intentions(input_context)
    
    # Navigation phylogénétique guidée
    conceptual_path = self.navigate_conceptual_tree(relevant_intentions)
    
    # Guidance spécialisée par module
    guidance = self.generate_module_guidance(module_type, conceptual_path)
    
    return guidance
```

### Mode Validation Post-Génération
```python
def validate_post_generation(self, generated_output, original_intention):
    """
    Validation post-génération de la cohérence intentionnelle
    """
    # Score de cohérence phylogénétique
    phylo_likelihood = self.compute_phylogenetic_likelihood(
        generated_output, original_intention
    )
    
    # Alignement intentionnel
    intention_alignment = self.compute_intention_alignment(
        generated_output, original_intention
    )
    
    # Validation composite
    validation_score = (phylo_likelihood + intention_alignment) / 2
    
    return {
        'approved': validation_score > self.validation_threshold,
        'confidence': validation_score,
        'phylo_likelihood': phylo_likelihood,
        'intention_alignment': intention_alignment
    }
```

## Métriques de performance

### Métriques phylogénétiques
- **Bootstrap Confidence** : Confiance statistique intentions (>70%)
- **Robinson-Foulds Distance** : Distance entre arbres conceptuels
- **Phylogenetic Likelihood** : Vraisemblance conceptuelle
- **SPR Prediction Accuracy** : Précision prédicteur SPR (>80%)

### Métriques intentionnelles
- **Refinement Quality** : Qualité raffinement méta → full intentions
- **Conceptual Positioning** : Précision positionnement phylogénétique
- **Guidance Effectiveness** : Efficacité guidance modules avancés
- **Validation Accuracy** : Précision validation post-génération

## Intégration avec l'écosystème

### Interface MetaLIMEN → FullLIMEN
```python
def initialize_from_meta_limen(self, meta_limen_checkpoint, sft_model):
    """
    Initialisation FullLIMEN depuis MetaLIMEN
    """
    # Chargement méta-intentions
    meta_intentions = load_meta_intentions(meta_limen_checkpoint)
    
    # Raffinement avec capacités SFT
    refined_intentions = self.refine_meta_intentions(
        meta_intentions, sft_model
    )
    
    # Construction espace phylogénétique conceptuel
    self.conceptual_space = self.build_phylogenetic_space(
        refined_intentions
    )
    
    return refined_intentions
```

### Interface FullLIMEN → Modules
```python
def provide_conceptual_guidance(self, module_name, task_context):
    """
    Fourniture guidance conceptuelle aux modules avancés
    """
    guidance = {
        'conceptual_intentions': self.current_intentions,
        'phylogenetic_path': self.get_conceptual_path(task_context),
        'exploration_constraints': self.get_exploration_bounds(),
        'validation_criteria': self.get_validation_criteria()
    }
    
    return guidance
```

## Roadmap

- [x] **v1.0** : Raffinement intentions MetaLIMEN → FullLIMEN
- [x] **v1.1** : Espace phylogénétique conceptuel basique
- [ ] **v2.0** : ML-guided SPR conceptuel complet
- [ ] **v2.1** : Validation bootstrap phylogénétique
- [ ] **v3.0** : Multi-arbres conceptuels simultanés
- [ ] **v3.1** : Optimisation continue espace conceptuel 