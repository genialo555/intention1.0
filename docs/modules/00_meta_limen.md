# Module MetaLIMEN (Meta-Latent Intent Meta-Emergent Nexus)

## Vue d'ensemble

MetaLIMEN constitue le **point de départ intentionnel** du système, définissant les intentions d'apprentissage pré-linguistiques qui guideront toute la construction du système. Il résout le paradoxe bootstrap en utilisant des embeddings simples pour encoder des objectifs conceptuels de haut niveau avant toute compréhension linguistique complexe.

## Objectifs

1. **Définition intentions pré-linguistiques** : Encoder des objectifs d'apprentissage sans base linguistique complexe
2. **Espace méta-intentionnel** : Créer un espace conceptuel simple pour organiser les intentions
3. **Guidance SFT** : Orienter l'apprentissage linguistique selon les intentions définies
4. **Bootstrap résolution** : Permettre la définition d'intentions avant capacités linguistiques

## Position dans le Pipeline

```
**MetaLIMEN** → SFT_Guided → FullLIMEN → [ICM, RND, Transformer², MCTS] → Conceptual-GRPO
     ↓             ↓           ↓              ↓                            ↓
Meta-Intent   Intentional   Sophisticated   Conceptual                Phylogenetic
Definition    Learning      Intentions      Modules                   Optimization
```

MetaLIMEN est le **premier module** qui s'exécute, définissant les intentions qui guideront tout le système.

## Architecture fonctionnelle

### Composants principaux

| Composant | Fonction |
|-----------|----------|
| **Simple Embedder** | Embeddings basiques (Word2Vec) pour concepts simples |
| **Meta-Intention Space** | Espace vectoriel pour intentions de haut niveau |
| **Domain Mapper** | Mapping domaines → intentions d'apprentissage |
| **Priority Calculator** | Calcul priorités d'apprentissage par domaine |
| **Curriculum Generator** | Génération curriculum pour SFT guidé |

### Architecture simple

```python
class MetaLIMEN:
    """
    MetaLIMEN pour intentions pré-linguistiques
    Utilise des embeddings simples, pas de compréhension complexe
    """
    def __init__(self):
        # Embeddings simples (Word2Vec, GloVe)
        self.simple_embedder = SimpleWordEmbedder()
        
        # Espace méta-intentionnel
        self.meta_intention_space = MetaIntentionSpace(dim=64)
        
        # Mapping domaines → intentions
        self.domain_mapper = DomainIntentionMapper()
        
        # Générateur curriculum
        self.curriculum_generator = SimpleCurriculumGenerator()
        
    def define_learning_intentions(self, domain_descriptions):
        """
        Définit des intentions d'apprentissage simples pour chaque domaine
        """
        meta_intentions = []
        
        for domain in domain_descriptions:
            # Embedding simple du domaine
            domain_embedding = self.simple_embedder.embed(domain)
            
            # Positionnement dans l'espace méta-intentionnel
            meta_position = self.meta_intention_space.locate(domain_embedding)
            
            # Création méta-intention
            meta_intention = {
                'domain': domain,
                'embedding': domain_embedding,
                'meta_position': meta_position,
                'learning_priority': self.calculate_priority(meta_position),
                'curriculum_weight': self.calculate_curriculum_weight(domain)
            }
            
            meta_intentions.append(meta_intention)
        
        return meta_intentions
```

## Configuration

```yaml
# === Architecture simple ===
embedding_dim: 64                # Dimension embeddings simples
meta_space_dim: 64              # Dimension espace méta-intentionnel
simple_embedder: "word2vec"     # Type embeddings simples

# === Domaines cibles ===
target_domains:
  - name: "quantum_computing"
    description: "design and analysis of algorithms on quantum machines"
    priority: 0.85
    curriculum_weight: 0.25
  - name: "synthetic_biology"
    description: "engineering biological systems for novel functions"
    priority: 0.75
    curriculum_weight: 0.15
  - name: "behavioral_economics"
    description: "psychological influences on economic decision-making"
    priority: 0.7
    curriculum_weight: 0.15
  - name: "cognitive_neuroscience"
    description: "neural mechanisms underlying cognition and behavior"
    priority: 0.8
    curriculum_weight: 0.2
  - name: "climate_modeling"
    description: "simulation and prediction of Earth's climate systems"
    priority: 0.65
    curriculum_weight: 0.1
  - name: "complex_adaptive_systems"
    description: "dynamics of systems with multiple interacting agents"
    priority: 0.6
    curriculum_weight: 0.15

# === Curriculum génération ===
curriculum_generation:
  data_filtering: true           # Filtrage données selon intentions
  priority_weighting: true      # Pondération selon priorités
  progression_mapping: true     # Mapping progression apprentissage

# === Méta-apprentissage ===
meta_learning:
  intention_evolution: false    # Pas d'évolution complexe (trop tôt)
  simple_adaptation: true       # Adaptation simple basée usage
  feedback_integration: false   # Pas de feedback complexe
```

## Pipeline d'entraînement

### 1. Préparation corpus conceptuels
```bash
python scripts/prepare_conceptual_data.py \
  --domains "physics,biology,economics,psychology" \
  --output data/conceptual_corpus/ \
  --simple_embeddings word2vec
```

### 2. Entraînement MetaLIMEN
```bash
python scripts/train_meta_limen.py \
  --config configs/meta_limen_config.yaml \
  --conceptual_data data/conceptual_corpus/ \
  --output models/meta_limen/
```

### 3. Génération intentions pour SFT
```bash
python scripts/generate_sft_intentions.py \
  --meta_limen_checkpoint models/meta_limen/latest.pt \
  --output models/meta_limen/intentions.pt
```

## Fonctionnement détaillé

### 1. Définition intentions par domaine
```python
def define_domain_intentions(self, domain_descriptions):
    """
    Définit les intentions d'apprentissage pour chaque domaine
    """
    intentions = {}
    
    for domain, description in domain_descriptions.items():
        # Embedding simple description domaine
        domain_vector = self.simple_embedder.embed(description)
        
        # Positionnement dans espace méta-intentionnel
        meta_position = self.meta_intention_space.map_domain_to_intention(
            domain_vector
        )
        
        # Création intention structurée
        intention = {
            'domain': domain,
            'description': description,
            'vector': domain_vector,
            'meta_position': meta_position,
            'learning_objectives': self.extract_learning_objectives(description),
            'curriculum_priorities': self.calculate_curriculum_priorities(domain)
        }
        
        intentions[domain] = intention
    
    return intentions
```

### 2. Génération curriculum SFT
```python
def generate_sft_curriculum(self, meta_intentions):
    """
    Génère un curriculum pour SFT basé sur les méta-intentions
    """
    curriculum = {
        'data_filtering_criteria': {},
        'priority_weights': {},
        'progression_stages': []
    }
    
    for domain, intention in meta_intentions.items():
        # Critères filtrage données
        curriculum['data_filtering_criteria'][domain] = \
            self.create_data_filter_criteria(intention)
        
        # Poids priorités apprentissage
        curriculum['priority_weights'][domain] = intention['curriculum_priorities']
        
        # Étapes progression
        progression = self.create_progression_stages(intention)
        curriculum['progression_stages'].extend(progression)
    
    return curriculum
```

### 3. Interface avec SFT guidé
```python
def provide_sft_guidance(self, training_data, meta_intentions):
    """
    Fournit guidance pour SFT basée sur méta-intentions
    """
    guidance = {
        'filtered_data': {},
        'loss_weights': {},
        'curriculum_schedule': []
    }
    
    # Filtrage données par domaine d'intention
    for domain, intention in meta_intentions.items():
        domain_data = self.filter_data_by_intention(training_data, intention)
        guidance['filtered_data'][domain] = domain_data
        
        # Poids loss selon priorité intention
        guidance['loss_weights'][domain] = intention['learning_priority']
    
    # Programmation curriculum
    guidance['curriculum_schedule'] = self.create_curriculum_schedule(
        meta_intentions
    )
    
    return guidance
```

## Métriques de performance

### Métriques méta-intentionnelles
- **Intention Coherence** : Cohérence des intentions définies (cosine similarity > 0.7 entre domaines différents)
- **Coverage Completeness** : Couverture complète domaines cibles (4/4 domaines)
- **Priority Consistency** : Consistance des priorités définies
- **Embedding Quality** : Qualité embeddings simples (voisins cohérents)

### Métriques curriculum
- **Filtering Effectiveness** : Efficacité filtrage données (% données alignées)
- **Weight Optimization** : Optimisation poids apprentissage
- **Progression Logic** : Logique progression curriculum
- **SFT Guidance Impact** : Impact guidance sur performance SFT

### Métriques simplicité
- **Computational Efficiency** : Efficacité computationnelle (< 1 min training)
- **Memory Usage** : Usage mémoire minimal (< 1GB)
- **Interpretation Clarity** : Clarté interprétation intentions
- **Bootstrap Success** : Succès résolution paradoxe bootstrap

## Exemples d'intentions générées

### Physique
```python
physics_intention = {
    'domain': 'physics',
    'description': 'mathematical physics concepts reasoning',
    'meta_position': [0.8, 0.6, 0.9, 0.7],  # Vecteur 64D simplifié
    'learning_objectives': [
        'mathematical_reasoning',
        'physical_intuition', 
        'quantitative_analysis',
        'formula_manipulation'
    ],
    'curriculum_priorities': {
        'mathematics': 0.4,
        'physics_concepts': 0.3,
        'problem_solving': 0.3
    },
    'data_filter_criteria': {
        'keywords': ['physics', 'mathematical', 'equation', 'formula'],
        'complexity_level': 'intermediate',
        'reasoning_type': 'quantitative'
    }
}
```

### Biologie
```python
biology_intention = {
    'domain': 'biology',
    'description': 'biological systems understanding',
    'meta_position': [0.6, 0.8, 0.5, 0.9],
    'learning_objectives': [
        'systems_thinking',
        'biological_processes',
        'life_sciences_reasoning',
        'classification_skills'
    ],
    'curriculum_priorities': {
        'biological_concepts': 0.5,
        'systems_analysis': 0.3,
        'classification': 0.2
    },
    'data_filter_criteria': {
        'keywords': ['biology', 'organism', 'cell', 'evolution'],
        'complexity_level': 'intermediate',
        'reasoning_type': 'systemic'
    }
}
```

## Validation

### Tests de cohérence intentionnelle
```python
def test_intention_coherence(meta_intentions):
    """
    Test cohérence des intentions générées
    """
    # Distances entre domaines différents
    for domain1, intention1 in meta_intentions.items():
        for domain2, intention2 in meta_intentions.items():
            if domain1 != domain2:
                distance = cosine_distance(
                    intention1['meta_position'],
                    intention2['meta_position']
                )
                assert distance > 0.3  # Domaines différents bien séparés
    
    # Cohérence interne chaque domaine
    for domain, intention in meta_intentions.items():
        assert len(intention['learning_objectives']) >= 3
        assert sum(intention['curriculum_priorities'].values()) == 1.0
```

### Tests de guidance SFT
```python
def test_sft_guidance_effectiveness(guidance, test_data):
    """
    Test efficacité guidance pour SFT
    """
    # Vérification filtrage données
    for domain, filtered_data in guidance['filtered_data'].items():
        # Données filtrées pertinentes au domaine
        relevance_score = compute_domain_relevance(filtered_data, domain)
        assert relevance_score > 0.8
    
    # Vérification équilibre poids
    total_weight = sum(guidance['loss_weights'].values())
    assert abs(total_weight - 1.0) < 0.1
```

## Intégration avec SFT

### Interface MetaLIMEN → SFT
```python
class MetaLIMENToSFTInterface:
    """
    Interface entre MetaLIMEN et SFT guidé
    """
    def __init__(self, meta_limen_checkpoint):
        self.meta_limen = load_meta_limen(meta_limen_checkpoint)
        
    def prepare_intentional_sft_data(self, raw_training_data):
        """
        Prépare données SFT selon intentions MetaLIMEN
        """
        meta_intentions = self.meta_limen.get_current_intentions()
        
        # Filtrage et pondération selon intentions
        prepared_data = self.meta_limen.provide_sft_guidance(
            raw_training_data, meta_intentions
        )
        
        return prepared_data
        
    def get_curriculum_schedule(self):
        """
        Obtient programmation curriculum selon intentions
        """
        return self.meta_limen.get_curriculum_schedule()
```

## Limitations et scope

### Ce que MetaLIMEN fait
✅ **Définit intentions simples** sans base linguistique complexe  
✅ **Guide SFT** selon objectifs conceptuels de haut niveau  
✅ **Résout bootstrap** : intentions avant capacités linguistiques  
✅ **Curriculum simple** basé sur priorités de domaines  

### Ce que MetaLIMEN ne fait pas
❌ **Compréhension linguistique complexe** (rôle de SFT + FullLIMEN)  
❌ **Validation temps réel** (rôle de FullLIMEN)  
❌ **Apprentissage adaptatif** (trop complexe pour ce stade)  
❌ **Interaction avec modules avancés** (rôle de FullLIMEN)  

## Roadmap

- [x] **v1.0** : Définition intentions simples par domaine
- [x] **v1.1** : Génération curriculum SFT basique
- [ ] **v2.0** : Optimisation embeddings simples
- [ ] **v2.1** : Affinement filtrage données
- [ ] **v3.0** : Adaptation légère basée sur feedback SFT
- [ ] **v3.1** : Extension à nouveaux domaines 