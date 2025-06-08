# Domain Mapping with MetaLIMEN

In this notebook we'll show how MetaLIMEN embeds domain descriptions into a meta-intention space, computes inter-domain distances, and visualizes their clustering.

```python
import yaml, json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from modules.meta_limen.meta_limen import MetaLIMEN
```

## Load configuration & instantiate MetaLIMEN

```python
cfg_file = "configs/meta_limen_config.yaml"
ml = MetaLIMEN(cfg_file)
cfg = yaml.safe_load(open(cfg_file))
domains = [d["name"] for d in cfg["target_domains"]]
```

## Define meta-intentions & build DataFrame

```python
intents = ml.define_learning_intentions()
df = pd.DataFrame([{
    "domain": i["domain"],
    **{f"dim{j}": v for j, v in enumerate(i["meta_position"])}
}
 for i in intents])
```

## Inter-domain Cosine Similarity Heatmap

```python
M = cosine_similarity(df.filter(regex='^dim').values)
sns.heatmap(M, xticklabels=df.domain, yticklabels=df.domain, cmap="viridis", annot=True)
plt.title("Inter-Domain Cosine Similarities")
plt.show()
```

## PCA of Meta-Intentions

```python
pca_results = PCA(n_components=2).fit_transform(df[[c for c in df if c.startswith("dim")]])
fig, ax = plt.subplots()
ax.scatter(pca_results[:,0], pca_results[:,1])
for x, y, label in zip(pca_results[:,0], pca_results[:,1], df.domain):
    ax.text(x+0.01, y+0.01, label)
ax.set_title("PCA of Meta-Intentions")
plt.show()
```

## Interpretation

Notice how some domains cluster closer, indicating shared curricular resources or overlapping conceptual space. 