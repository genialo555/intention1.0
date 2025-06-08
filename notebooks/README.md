# Notebooks

Cette directory contient des démonstrations interactives en Jupyter Notebook pour :

- `domain_mapping_demo.ipynb` : Visualisation et exploration de la normalisation des vecteurs d'intention (DomainIntentionMapper)
- `semantic_filtering_demo.ipynb` : Exécution pas-à-pas des critères de filtrage sémantique et des stages de progression (SimpleCurriculumGenerator)

Chaque notebook suit le plan :
1. Chargement de la configuration `configs/meta_limen_config.yaml`
2. Instanciation des classes MetaLIMEN et des générateurs
3. Affichage des résultats (vecteurs normalisés, critères, steps)
4. Visualisations (histogrammes, heatmaps) pour illustrer la distribution des embeddings

Pour lancer ces notebooks :

```bash
source .venv/bin/activate
jupyter lab notebooks/
``` 