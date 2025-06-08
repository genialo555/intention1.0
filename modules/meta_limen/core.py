import torch
import torch.nn as nn
import torch.nn.functional as F


class MetaLimenCore(nn.Module):
    """
    Core MetaLIMEN model for training embeddings with coherence and separation losses.
    """
    def __init__(self, input_dim: int = 64, meta_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, meta_dim)
        )
        self.meta_dim = meta_dim
        # default margins for loss
        self.coherence_margin = 0.0  # minimum acceptable similarity for same-class
        self.separation_margin = 0.2  # maximum acceptable similarity for diff-class

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute latent embeddings for input tensor x.
        """
        return self.encoder(x)

    def loss(self,
             representations: torch.Tensor,
             labels: torch.Tensor,
             separation_weight: float = 0.3,
             coherence_weight: float = 0.7,
             reg_weight: float = 0.1,
             separation_margin: float = None,
             coherence_margin: float = None
             ) -> (torch.Tensor, dict):
        """
        Compute total loss as weighted sum of coherence, separation, and regularization.

        Args:
            representations: [batch, meta_dim] tensor of embeddings.
            labels: [batch] tensor of integer labels.
            separation_weight: weight for inter-class separation loss.
            coherence_weight: weight for intra-class coherence loss.
            reg_weight: weight for embedding regularization.
            separation_margin: optional override of diff-class margin.
            coherence_margin: optional override of same-class margin.

        Returns:
            total loss tensor and a dict of individual loss values.
        """
        # Normalize embeddings
        reps = F.normalize(representations, p=2, dim=-1)

        # Similarity matrix
        sim_matrix = reps @ reps.t()
                
        # Choose margins
        sep_margin = separation_margin if separation_margin is not None else self.separation_margin
        coh_margin = coherence_margin if coherence_margin is not None else self.coherence_margin
        
        # Coherence: penalize if same-class similarity below (1 - coh_margin)
        same_class = labels.unsqueeze(0) == labels.unsqueeze(1)
        # target similarity = 1.0, penalize hinge: max(0, target - sim - coh_margin)
        coh_diff = (1.0 - sim_matrix[same_class]) - coh_margin
        coherence_loss = torch.clamp(coh_diff, min=0).mean()

        # Separation: penalize if diff-class similarity above sep_margin
        diff_class = ~same_class
        sep_diff = sim_matrix[diff_class] - sep_margin
        separation_loss = torch.clamp(sep_diff, min=0).mean()

        # Regularization: keep norms small
        reg = (reps.norm(p=2, dim=-1) ** 2).mean()

        total = (
            coherence_weight * coherence_loss
            + separation_weight * separation_loss
            + reg_weight * reg
        )
        metrics = {
            "coherence": coherence_loss.item(),
            "separation": separation_loss.item(),
            "reg": reg.item(),
            "total": total.item()
        }
        return total, metrics 