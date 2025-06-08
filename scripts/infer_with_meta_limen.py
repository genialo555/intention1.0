#!/usr/bin/env python3
import yaml
from modules.meta_limen.meta_limen import MetaLIMEN
from transformers import pipeline
import torch

def main():
    # Load MetaLIMEN config and define learning intentions
    config_path = "configs/meta_limen_config.yaml"
    ml = MetaLIMEN(config_path)
    meta_intentions = ml.define_learning_intentions()
    # Use the first intention as prefix
    intent = meta_intentions[0]
    # Prompt user for a question
    user_prompt = input("Question> ")
    # Enrich prompt with intention description
    enriched_prompt = f"Intention d'apprentissage: {intent['description']}\nQuestion: {user_prompt}"
    # Initialize text-generation pipeline
    pipe = pipeline(
        "text-generation",
        model="models/base/qwen2.5-math-7b",
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16
    )
    # Generate and print output
    result = pipe(enriched_prompt, max_new_tokens=128, temperature=0.2)
    print(result[0]["generated_text"])

if __name__ == "__main__":
    main() 