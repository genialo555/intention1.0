#!/usr/bin/env python3
"""
Script to train the MetaLimenCore model on pre-linguistic domain descriptions.
"""
import os
import sys
import argparse
import yaml
import json
import torch
from torch.utils.data import DataLoader, TensorDataset

# ensure project root on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.meta_limen.core import MetaLimenCore
from modules.meta_limen.meta_limen import SimpleWordEmbedder
from modules.meta_limen.embedder import DeepSeekEmbedder


def main():
    parser = argparse.ArgumentParser(description="Train MetaLimenCore on domain descriptions.")
    parser.add_argument('--config', type=str, required=True, help='Path to meta_limen config YAML')
    parser.add_argument('--output', type=str, required=True, help='Output directory for model and metrics')
    args = parser.parse_args()

    # load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # training hyperparameters
    train_cfg = config.get('training', {})
    epochs = int(train_cfg.get('max_epochs', 5))
    lr = float(train_cfg.get('learning_rate', 1e-3))
    batch_size = int(train_cfg.get('batch_size', 32))

    # loss weights/margins
    opt_cfg = train_cfg.get('intention_optimization', {})
    sep_w = float(opt_cfg.get('separation_loss_weight', 0.3))
    coh_w = float(opt_cfg.get('coherence_loss_weight', 0.7))
    reg_w = float(opt_cfg.get('regularization_weight', 0.1))
    margin_cfg = config.get('meta_intention_space', {})
    sep_margin = float(margin_cfg.get('separation_penalty', 0.2))
    # keep coherence margin at zero (we want sim=1)
    coh_margin = float(margin_cfg.get('coherence_reward', 0.0))

    # prepare dataset from config target_domains
    domains = config.get('target_domains', [])
    descriptions = [d.get('description', '') for d in domains]
    labels = list(range(len(domains)))

    # choose embedder based on config
    cfg_embed = str(config.get('simple_embedder', 'word2vec')).lower()
    if cfg_embed == 'deepseek':
        ds_path = config.get('pretrained_embeddings')
        embedder = DeepSeekEmbedder(ds_path)
    else:
        embedder = SimpleWordEmbedder(dim=int(config.get('embedding_dim', 64)))
    # determine actual embedding dimension
    embed_dim = embedder.dim
    print(f"[DEBUG] Using embedder dimension: {embed_dim}")
    # embed descriptions
    embeds = [embedder.embed(desc) for desc in descriptions]
    X = torch.tensor(embeds, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long)
    print(f"[DEBUG] X tensor shape before device move: {X.shape}")

    # move tensors to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Training on device: {device}")
    X = X.to(device)
    y = y.to(device)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # init model, optimizer
    model = MetaLimenCore(input_dim=embed_dim, meta_dim=int(config.get('meta_space_dim', embed_dim)))
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ready output dir
    os.makedirs(args.output, exist_ok=True)

    history = {'epoch': [], 'coherence': [], 'separation': [], 'reg': [], 'total': []}

    # training loop
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_metrics = {'coherence': 0.0, 'separation': 0.0, 'reg': 0.0, 'total': 0.0}
        count = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            reps = model(xb)
            loss, metrics = model.loss(
                reps, yb,
                separation_weight=sep_w,
                coherence_weight=coh_w,
                reg_weight=reg_w,
                separation_margin=sep_margin,
                coherence_margin=coh_margin
            )
            loss.backward()
            optimizer.step()
            # accumulate metrics
            for k in epoch_metrics:
                epoch_metrics[k] += metrics[k]
            count += 1
        # average
        for k in epoch_metrics:
            epoch_metrics[k] /= count
        history['epoch'].append(epoch)
        for k in epoch_metrics:
            history[k].append(epoch_metrics[k])
        print(f"Epoch {epoch}/{epochs}: total={epoch_metrics['total']:.4f}, coh={epoch_metrics['coherence']:.4f}, sep={epoch_metrics['separation']:.4f}, reg={epoch_metrics['reg']:.4f}")

    # save model
    model_path = os.path.join(args.output, 'meta_limen_core.pt')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # save metrics
    metrics_path = os.path.join(args.output, 'training_history.json')
    with open(metrics_path, 'w') as mf:
        json.dump(history, mf, indent=2)
    print(f"Training history saved to {metrics_path}")


if __name__ == '__main__':
    main() 