#!/usr/bin/env python3
import os
import sys

# Ensure imports work from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.config import PipelineConfig
from data.dataset import load_dataset
from data.normalization import Normalizer
from models.config import TransformerConfig
from training.transformer_trainer import TrainingConfig
from training.transformer_trainer import train
from scripts.generate_data import generate_synthetic_jsonl

def main():
    pipeline_cfg = PipelineConfig()
    
    if not os.path.exists(pipeline_cfg.raw_data_path):
        print(f"Generating synthetic data at {pipeline_cfg.raw_data_path}...")
        generate_synthetic_jsonl(pipeline_cfg.raw_data_path, 2000)
        
    print("Loading synthetic data and generating normalized dataset...")
    dataset = load_dataset(pipeline_cfg)
    
    feat_norm = Normalizer().load(pipeline_cfg.normalization_path)
    tgt_norm_path = pipeline_cfg.normalization_path.replace(".json", "_targets.json")
    tgt_norm = Normalizer().load(tgt_norm_path)
    
    model_cfg = TransformerConfig()
    training_cfg = TrainingConfig(
        epochs=15, 
        batch_size=16, 
        lr=1e-3, 
        checkpoint_dir="checkpoints"
    )
    
    os.makedirs(training_cfg.checkpoint_dir, exist_ok=True)
    
    print("Training Transformer...")
    best_model, ckpt_path = train(
        pipeline_cfg, model_cfg, training_cfg,
        windows=dataset,
        feat_normalizer=feat_norm,
        tgt_normalizer=tgt_norm,
    )
    print(f"Transformer training complete. Best model saved to: {ckpt_path}")

if __name__ == '__main__':
    main()
