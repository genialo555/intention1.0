#!/usr/bin/env python3
"""
Script to download the DeepSeek r1 Qwen 12B model as our base model.
"""
import os
from huggingface_hub import snapshot_download

MODEL_ID = "Qwen/Qwen2.5-Math-7B"
OUTPUT_DIR = os.path.join("models", "base", "qwen2.5-math-7b")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Downloading full repository {MODEL_ID}...")
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=OUTPUT_DIR,
        local_dir_use_symlinks=False,
        use_auth_token=True  # ensures access if needed
    )
    print(f"Base model downloaded to {OUTPUT_DIR}")


if __name__ == "__main__":
    main() 