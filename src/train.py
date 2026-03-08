"""
Training script for fine-tuning the bi-encoder model with contrastive loss.

Usage:
    python src/train.py --epochs 5 --batch_size 16 --model_name all-MiniLM-L6-v2
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import SentenceEvaluator
from torch.utils.data import DataLoader
from tqdm import tqdm
import math

from dataset import (
    load_train,
    train_val_split,
    create_contrastive_pairs,
    oversample_rare_labels,
    get_label_set,
)
from evaluate import map_at_k
from model import MisconceptionBiEncoder, retrieve_top_k


def create_training_examples(input_texts, labels):
    """
    Create InputExample objects for sentence-transformers training.
    
    For MultipleNegativesRankingLoss, each example is (text, label).
    The loss assumes that within a batch, each (text, label) pair is positive,
    and all other labels in the batch are negatives for that text.
    """
    examples = []
    for text, label in zip(input_texts, labels):
        examples.append(InputExample(texts=[text, label]))
    return examples


class CustomEvaluator(SentenceEvaluator):
    """Custom evaluator for MAP@3 metric during training."""
    
    def __init__(self, model: MisconceptionBiEncoder, val_df: pd.DataFrame, train_df: pd.DataFrame, k: int = 3):
        self.model = model
        self.val_df = val_df
        self.train_df = train_df
        self.k = k
        
    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        """Compute MAP@3 on validation set."""
        # Get unique labels from training set
        unique_labels = sorted(self.train_df["full_label"].unique().tolist())
        
        # Encode label prototypes
        label_embeddings = self.model.encode(
            unique_labels,
            batch_size=64,
            show_progress=False,
            normalize=True,
            convert_to_numpy=True
        )
        
        # Encode validation inputs
        val_inputs = self.val_df["model_input"].tolist()
        val_embeddings = self.model.encode(
            val_inputs,
            batch_size=64,
            show_progress=False,
            normalize=True,
            convert_to_numpy=True
        )
        
        # Retrieve top-k predictions
        predictions = retrieve_top_k(
            val_embeddings,
            label_embeddings,
            unique_labels,
            k=self.k
        )
        
        # Compute MAP@k
        y_true = self.val_df["full_label"].tolist()
        overall_map, _ = map_at_k(y_true, predictions, k=self.k)
        
        return overall_map


def evaluate_model(
    model: MisconceptionBiEncoder,
    val_df: pd.DataFrame,
    train_df: pd.DataFrame,
    k: int = 3
) -> tuple[float, dict]:
    """
    Evaluate the model on validation set using MAP@k.
    
    Strategy:
    1. Encode all unique labels from training set (label prototypes)
    2. Encode all validation inputs
    3. For each val input, find top-k nearest label prototypes
    4. Compute MAP@k
    
    Args:
        model: The bi-encoder model
        val_df: Validation DataFrame
        train_df: Training DataFrame (to get label prototypes)
        k: Number of predictions (default 3)
        
    Returns:
        Tuple of (overall_map, per_label_map_dict)
    """
    print("\n=== Running validation ===")
    
    # Get unique labels from training set
    unique_labels = sorted(train_df["full_label"].unique().tolist())
    print(f"Encoding {len(unique_labels)} unique label prototypes...")
    
    # Encode label prototypes
    label_embeddings = model.encode(
        unique_labels,
        batch_size=64,
        show_progress=False,
        normalize=True,
        convert_to_numpy=True
    )
    
    # Encode validation inputs
    print(f"Encoding {len(val_df)} validation examples...")
    val_inputs = val_df["model_input"].tolist()
    val_embeddings = model.encode(
        val_inputs,
        batch_size=64,
        show_progress=False,
        normalize=True,
        convert_to_numpy=True
    )
    
    # Retrieve top-k predictions
    print("Computing top-k predictions...")
    predictions = retrieve_top_k(
        val_embeddings,
        label_embeddings,
        unique_labels,
        k=k
    )
    
    # Compute MAP@k
    y_true = val_df["full_label"].tolist()
    overall_map, per_label_map = map_at_k(y_true, predictions, k=k)
    
    print(f"Validation MAP@{k}: {overall_map:.4f}")
    
    return overall_map, per_label_map


def train(
    data_path: str = "data/train_clean.csv",
    model_name: str = "all-MiniLM-L6-v2",
    output_dir: str = "models",
    num_epochs: int = 4,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    warmup_steps: int = 100,
    val_frac: float = 0.2,
    oversample: bool = True,
    oversample_min: int = 10,
    seed: int = 42,
    train_df: pd.DataFrame = None,
    val_df: pd.DataFrame = None,
    run_name: str = None,
):
    """
    Main training function.

    If train_df and val_df are provided, uses those directly instead of
    loading from data_path. This is useful for ablation experiments where
    the data is pre-processed differently.
    """
    print("=" * 60)
    print("Phase 3: Fine-Tuning Bi-Encoder with Contrastive Loss")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Model: {model_name}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Oversample: {oversample} (min={oversample_min})")
    if torch.cuda.is_available():
        device_name = "CUDA"
    elif torch.backends.mps.is_available():
        device_name = "MPS (Apple Silicon)"
    else:
        device_name = "CPU"
    print(f"  Device: {device_name}")

    # Load and split data (or use provided DataFrames)
    if train_df is not None and val_df is not None:
        print(f"\n[1/6] Using provided train/val DataFrames...")
        print(f"  Train: {len(train_df)} | Val: {len(val_df)}")
    else:
        print(f"\n[1/6] Loading data from {data_path}...")
        df = load_train(data_path)
        print(f"  Total examples: {len(df)}")
        print(f"  Unique labels: {df['full_label'].nunique()}")
        train_df, val_df = train_val_split(df, val_frac=val_frac, seed=seed)
        print(f"  Train: {len(train_df)} | Val: {len(val_df)}")
    
    # Optionally oversample rare labels
    if oversample:
        print(f"\n[2/6] Oversampling rare labels (target min: {oversample_min})...")
        train_df_original_size = len(train_df)
        train_df = oversample_rare_labels(
            train_df,
            target_min_count=oversample_min,
            seed=seed
        )
        print(f"  Train size after oversampling: {len(train_df)} (+{len(train_df) - train_df_original_size})")
    else:
        print("\n[2/6] Skipping oversampling")
    
    # Create training pairs
    print("\n[3/6] Creating contrastive training pairs...")
    input_texts, labels = create_contrastive_pairs(train_df)
    train_examples = create_training_examples(input_texts, labels)
    print(f"  Created {len(train_examples)} training examples")
    
    # Initialize model
    print(f"\n[4/6] Loading base model: {model_name}...")
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    model = MisconceptionBiEncoder(model_name=model_name, device=device)
    base_model = model.get_base_model()
    
    print(f"  Embedding dimension: {model.embedding_dim}")
    print(f"  Device: {device}")
    
    # Define loss function
    print("\n[5/6] Setting up training with MultipleNegativesRankingLoss...")
    train_loss = losses.MultipleNegativesRankingLoss(base_model)
    
    # Output directory setup
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if run_name:
        run_dir = output_path / run_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = output_path / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output directory: {run_dir}")
    
    # Save config
    config = {
        "model_name": model_name,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "warmup_steps": warmup_steps,
        "train_size": len(train_df),
        "val_size": len(val_df),
        "num_labels": train_df["full_label"].nunique(),
        "oversample": oversample,
        "oversample_min": oversample_min,
        "seed": seed,
    }
    config_path = run_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved config to {config_path}")
    
    # Evaluate before training (baseline)
    print("\n[6/6] Evaluating before training (zero-shot)...")
    baseline_map, _ = evaluate_model(model, val_df, train_df, k=3)
    
    # Setup for manual training loop
    print("\n[Setting up training...]")
    
    # Create DataLoader with smart_batching_collate
    from sentence_transformers import util
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=base_model.smart_batching_collate
    )
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(base_model.parameters(), lr=learning_rate)
    
    # Calculate total steps for scheduler
    steps_per_epoch = len(train_dataloader)
    total_steps = steps_per_epoch * num_epochs
    
    # Create learning rate scheduler with warmup
    def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    # Training loop
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    
    best_map = baseline_map
    best_epoch = 0
    history = {
        "epochs": [],
        "val_map": [],
    }
    
    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*60}")
        
        # Train for one epoch
        base_model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}")
        for batch_features, batch_labels in progress_bar:
            # Forward pass - get embeddings
            sentence_features = []
            for features in batch_features:
                # Move features to device
                features = {key: value.to(device) for key, value in features.items()}
                sentence_features.append(features)
            
            # Compute loss
            loss_value = train_loss(sentence_features, labels=batch_labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss_value.item()
            num_batches += 1
            progress_bar.set_postfix({'loss': f'{loss_value.item():.4f}'})
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"Average training loss: {avg_loss:.4f}")
        
        # Evaluate on validation set
        val_map, per_label_map = evaluate_model(model, val_df, train_df, k=3)
        
        # Track history
        history["epochs"].append(epoch + 1)
        history["val_map"].append(val_map)
        
        # Save checkpoint if best
        if val_map > best_map:
            best_map = val_map
            best_epoch = epoch + 1
            best_model_path = run_dir / "best_model"
            model.save(best_model_path)
            print(f"\n✓ New best model! MAP@3={val_map:.4f} (saved to {best_model_path})")
            
            # Save per-label results
            per_label_df = pd.DataFrame([
                {"label": label, "map": map_val}
                for label, map_val in sorted(per_label_map.items(), key=lambda x: x[1], reverse=True)
            ])
            per_label_path = run_dir / f"per_label_epoch{epoch + 1}.csv"
            per_label_df.to_csv(per_label_path, index=False)
        
        print(f"Best MAP@3 so far: {best_map:.4f} (epoch {best_epoch})")
    
    # Save final model
    final_model_path = run_dir / "final_model"
    model.save(final_model_path)
    print(f"\n✓ Final model saved to {final_model_path}")
    
    # Save training history
    history_df = pd.DataFrame(history)
    history_path = run_dir / "training_history.csv"
    history_df.to_csv(history_path, index=False)
    print(f"✓ Training history saved to {history_path}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Baseline MAP@3: {baseline_map:.4f}")
    print(f"Best MAP@3:     {best_map:.4f} (epoch {best_epoch})")
    print(f"Improvement:    {best_map - baseline_map:.4f} ({(best_map / baseline_map - 1) * 100:.1f}%)")
    print(f"\nBest model: {run_dir / 'best_model'}")
    print(f"Final model: {run_dir / 'final_model'}")
    
    return model, history, best_map


def main():
    parser = argparse.ArgumentParser(description="Train misconception bi-encoder")
    parser.add_argument("--data_path", type=str, default="data/train_clean.csv")
    parser.add_argument("--model_name", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--output_dir", type=str, default="models")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--no_oversample", action="store_true")
    parser.add_argument("--oversample_min", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    train(
        data_path=args.data_path,
        model_name=args.model_name,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        val_frac=args.val_frac,
        oversample=not args.no_oversample,
        oversample_min=args.oversample_min,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
