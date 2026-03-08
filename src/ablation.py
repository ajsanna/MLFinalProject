"""
Phase 4: Ablation studies.

Runs four controlled experiments to figure out what actually drives performance:
  1. Input ablation — which parts of the student input matter most?
  2. Model size — does a bigger encoder help?
  3. Fine-tuning vs frozen — how much does contrastive fine-tuning add?
  4. Training data size — how much labeled data do we actually need?

All experiments use the same 80/20 stratified split and evaluate on MAP@3.
Results are saved to models/ablation_results.csv.

Usage:
    python src/ablation.py
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

from dataset import load_train, train_val_split
from train import train
from model import MisconceptionBiEncoder, retrieve_top_k
from evaluate import map_at_k


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_model_input(df, mode):
    """Reconstruct the model_input column using different subsets of the fields."""
    if mode == "full":
        # Q + A + E — this is the default from Phase 3
        return df["model_input"]
    elif mode == "explanation_only":
        return df["StudentExplanation_clean"]
    elif mode == "q_and_e":
        return (
            "Question: " + df["QuestionText_clean"]
            + " Explanation: " + df["StudentExplanation_clean"]
        )
    elif mode == "a_and_e":
        return (
            "Answer: " + df["MC_Answer"]
            + " Explanation: " + df["StudentExplanation_clean"]
        )
    else:
        raise ValueError(f"Unknown input mode: {mode}")


def subsample_train(train_df, fraction, seed=42):
    """
    Take a stratified subsample of the training data.
    Keeps at least 1 example per label so every class is represented.
    """
    if fraction >= 1.0:
        return train_df.copy()

    parts = []
    for label, group in train_df.groupby("full_label"):
        n = max(1, int(len(group) * fraction))
        parts.append(group.sample(n=n, random_state=seed))

    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Phase 4: Ablation Studies")
    print("=" * 60)

    # Use the same split for every experiment
    df = load_train("data/train_clean.csv")
    train_df, val_df = train_val_split(df, val_frac=0.2, seed=42)
    print(f"Loaded data: {len(train_df)} train / {len(val_df)} val")
    print(f"Unique labels: {train_df['full_label'].nunique()}\n")

    results = []

    # ------------------------------------------------------------------
    # Baseline run (full input, MiniLM, 100% data)
    # This single run serves as the reference for Studies 1, 2, and 4.
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Baseline: full input / MiniLM / 100% data")
    print("=" * 60)

    _, _, baseline_map = train(
        model_name="all-MiniLM-L6-v2",
        num_epochs=4,
        batch_size=16,
        train_df=train_df.copy(),
        val_df=val_df.copy(),
        run_name="ablation_baseline",
    )

    # Register the baseline result under each study it belongs to
    results.append({"study": "input_ablation", "condition": "Q+A+E (full)", "map3": baseline_map})
    results.append({"study": "model_size",     "condition": "MiniLM-L6-v2",  "map3": baseline_map})
    results.append({"study": "data_size",      "condition": "100%",           "map3": baseline_map})

    # ------------------------------------------------------------------
    # Study 1 — Input ablation
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Study 1: Input Ablation")
    print("=" * 60)

    input_variants = [
        ("explanation_only", "Explanation only"),
        ("q_and_e",          "Q+E"),
        ("a_and_e",          "A+E"),
    ]

    for mode, label in input_variants:
        print(f"\n--- {label} ---")

        train_mod = train_df.copy()
        val_mod = val_df.copy()
        train_mod["model_input"] = build_model_input(train_mod, mode)
        val_mod["model_input"] = build_model_input(val_mod, mode)

        _, _, best_map = train(
            model_name="all-MiniLM-L6-v2",
            num_epochs=4,
            batch_size=16,
            train_df=train_mod,
            val_df=val_mod,
            run_name=f"ablation_input_{mode}",
        )
        results.append({"study": "input_ablation", "condition": label, "map3": best_map})

    # ------------------------------------------------------------------
    # Study 2 — Model size
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Study 2: Model Size")
    print("=" * 60)

    print("\n--- MPNet-base ---")
    _, _, mpnet_map = train(
        model_name="all-mpnet-base-v2",
        num_epochs=4,
        batch_size=16,
        train_df=train_df.copy(),
        val_df=val_df.copy(),
        run_name="ablation_model_mpnet",
    )
    results.append({"study": "model_size", "condition": "MPNet-base-v2", "map3": mpnet_map})

    # ------------------------------------------------------------------
    # Study 3 — Fine-tuning vs frozen
    # No training needed here; just evaluate the frozen encoder.
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Study 3: Fine-tuning vs Frozen")
    print("=" * 60)

    frozen_model = MisconceptionBiEncoder("all-MiniLM-L6-v2")
    unique_labels = sorted(train_df["full_label"].unique().tolist())

    label_emb = frozen_model.encode(unique_labels, batch_size=64, normalize=True)
    val_emb = frozen_model.encode(
        val_df["model_input"].tolist(), batch_size=64, normalize=True
    )
    preds = retrieve_top_k(val_emb, label_emb, unique_labels, k=3)
    frozen_map, _ = map_at_k(val_df["full_label"].tolist(), preds, k=3)

    print(f"Frozen (zero-shot): MAP@3 = {frozen_map:.4f}")
    print(f"Fine-tuned:         MAP@3 = {baseline_map:.4f}")

    results.append({"study": "finetune_vs_frozen", "condition": "Frozen (zero-shot)", "map3": frozen_map})
    results.append({"study": "finetune_vs_frozen", "condition": "Fine-tuned",         "map3": baseline_map})

    # ------------------------------------------------------------------
    # Study 4 — Training data size
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Study 4: Training Data Size")
    print("=" * 60)

    for frac in [0.25, 0.50, 0.75]:
        pct = int(frac * 100)
        print(f"\n--- {pct}% of training data ---")

        train_sub = subsample_train(train_df, frac, seed=42)
        print(f"  Subsampled to {len(train_sub)} examples")

        _, _, sub_map = train(
            model_name="all-MiniLM-L6-v2",
            num_epochs=4,
            batch_size=16,
            train_df=train_sub,
            val_df=val_df.copy(),
            run_name=f"ablation_data_{pct}pct",
        )
        results.append({"study": "data_size", "condition": f"{pct}%", "map3": sub_map})

    # ------------------------------------------------------------------
    # Save + print results
    # ------------------------------------------------------------------
    results_df = pd.DataFrame(results)
    out_path = Path("models") / "ablation_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")

    print("\n" + "=" * 60)
    print("ABLATION RESULTS")
    print("=" * 60)

    for study in results_df["study"].unique():
        rows = results_df[results_df["study"] == study]
        print(f"\n{study}")
        print("-" * 40)
        for _, r in rows.iterrows():
            print(f"  {r['condition']:25s}  MAP@3 = {r['map3']:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
