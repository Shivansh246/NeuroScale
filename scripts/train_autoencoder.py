#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.config import PipelineConfig
from data.dataset import load_dataset
from data.normalization import Normalizer
from anomaly.config import AutoencoderConfig
from anomaly.trainer import train_autoencoder, AETrainingConfig
from scripts.generate_data import generate_synthetic_jsonl

def main():
    pipeline_cfg = PipelineConfig()
    
    if not os.path.exists(pipeline_cfg.raw_data_path):
        print(f"Generating synthetic data at {pipeline_cfg.raw_data_path}...")
        generate_synthetic_jsonl(pipeline_cfg.raw_data_path, 2000)
        
    print("Loading synthetic data and generating normalized dataset...")
    dataset = load_dataset(pipeline_cfg)
    
    feat_norm = Normalizer().load(pipeline_cfg.normalization_path)
    
    model_cfg = AutoencoderConfig()
    training_cfg = AETrainingConfig(
        epochs=15, 
        batch_size=16, 
        lr=1e-3, 
        checkpoint_dir="checkpoints"
    )
    
    os.makedirs(training_cfg.checkpoint_dir, exist_ok=True)
    
    print("Training Autoencoder...")
    best_model, ckpt_path = train_autoencoder(
        model_cfg, training_cfg,
        train_windows=dataset["train"],
        val_windows=dataset["val"],
        feat_normalizer=feat_norm
    )
    print(f"Autoencoder training complete. Best model saved to: {ckpt_path}")

if __name__ == '__main__':
    main()
