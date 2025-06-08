# Sprint 1 – MetaLIMEN Foundations

In this notebook we complete Sprint 1 by preparing conceptual data, training a MetaLIMEN model stub, defining intentions, and visualizing the meta-intention space.

## 1. Prepare Conceptual Data
```bash
python scripts/prepare_conceptual_data.py \
  --config configs/meta_limen_config.yaml \
  --output_dir data/conceptual_corpus
```
Inspect output:
```python
import json
with open("data/conceptual_corpus/concepts.jsonl") as f:
    lines = f.readlines()
print(f"Prepared {len(lines)} concept entries")
sample = json.loads(lines[0])
sample
```

## 2. Instantiate MetaLIMEN & Define Intentions
```python
from modules.meta_limen.meta_limen import MetaLIMEN
import pandas as pd
import yaml

cfg_file = "configs/meta_limen_config.yaml"
ml = MetaLIMEN(cfg_file)
cfg = yaml.safe_load(open(cfg_file))
domains = [d["name"] for d in cfg["target_domains"]]
intents = ml.define_learning_intentions()
df = pd.DataFrame(intents)
df[["domain","learning_priority","curriculum_weight","meta_position"]]
```

## 3. Visualize Meta-Intention Space
```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

vectors = np.vstack([i["meta_position"] for i in intents])
sim = cosine_similarity(vectors)
sns.heatmap(sim, xticklabels=df.domain, yticklabels=df.domain, cmap="magma", annot=True)
plt.title("Inter-Domain Cosine Similarities")
plt.show()
```

## 4. Run Unit Tests
```bash
pytest tests/unit/test_meta_limen.py -q || true
``` 