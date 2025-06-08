import torch
# monkey-patch for compatibility: ensure get_default_device exists
if not hasattr(torch, 'get_default_device'):
    torch.get_default_device = lambda *args, **kwargs: torch.device('cpu')
from transformers import AutoTokenizer, AutoModel
from typing import List

class DeepSeekEmbedder:
    """
    Embedder that uses a pretrained DeepSeek model to produce vector representations.
    """
    def __init__(self, model_name_or_path: str):
        # load tokenizer and model from HuggingFace
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model.eval()
        # record embedding dimension
        self.dim = self.model.config.hidden_size

    def embed(self, text: str) -> List[float]:
        # tokenize and run through model for embeddings
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # mean pooling over token embeddings
        last_hidden = outputs.last_hidden_state  # [1, seq_len, hidden]
        pooled = last_hidden.mean(dim=1).squeeze(0)  # [hidden]
        return pooled.tolist() 