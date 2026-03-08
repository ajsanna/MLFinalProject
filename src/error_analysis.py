"""
Phase 5: Error analysis on the best model (MPNet-base).

Generates:
  1. Category-level confusion matrix
  2. Most common misconception-level confusion pairs
  3. Per-label accuracy breakdown (best and worst performing labels)
  4. Example failure cases with student text

Usage:
    python src/error_analysis.py
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict

from dataset import load_train, train_val_split
from model import MisconceptionBiEncoder, retrieve_top_k
from evaluate import map_at_k, average_precision_at_k


MODEL_PATH = "models/ablation_model_mpnet/best_model"
DATA_PATH = "data/train_clean.csv"
OUTPUT_DIR = Path("models/error_analysis")


def get_predictions(model, val_df, train_df, k=3):
    """Run inference on the val set and return predictions + similarities."""
    unique_labels = sorted(train_df["full_label"].unique().tolist())

    print(f"Encoding {len(unique_labels)} label prototypes...")
    label_emb = model.encode(unique_labels, batch_size=64, normalize=True)

    print(f"Encoding {len(val_df)} validation examples...")
    val_emb = model.encode(
        val_df["model_input"].tolist(), batch_size=64, normalize=True
    )

    # Get similarity matrix for detailed analysis
    sim_matrix = val_emb @ label_emb.T

    # Top-k predictions per example
    predictions = []
    pred_scores = []
    for i in range(len(val_df)):
        top_idx = np.argsort(sim_matrix[i])[::-1][:k]
        predictions.append([unique_labels[j] for j in top_idx])
        pred_scores.append([float(sim_matrix[i][j]) for j in top_idx])

    return predictions, pred_scores, unique_labels


def category_confusion(val_df, predictions):
    """Build a confusion matrix at the Category level (True_Correct, etc.)."""
    true_cats = val_df["Category"].tolist()
    pred_cats = [p[0].split(":")[0] for p in predictions]  # top-1 prediction category

    cat_labels = sorted(set(true_cats) | set(pred_cats))
    matrix = pd.DataFrame(0, index=cat_labels, columns=cat_labels)

    for true, pred in zip(true_cats, pred_cats):
        matrix.loc[true, pred] += 1

    return matrix


def misconception_confusion_pairs(val_df, predictions, top_n=20):
    """
    Find the most common (true_label, predicted_label) error pairs.
    Only counts cases where the top-1 prediction is wrong.
    """
    error_pairs = Counter()
    true_labels = val_df["full_label"].tolist()

    for true_label, preds in zip(true_labels, predictions):
        top1 = preds[0]
        if top1 != true_label:
            error_pairs[(true_label, top1)] += 1

    return error_pairs.most_common(top_n)


def per_label_breakdown(val_df, predictions, k=3):
    """Compute MAP@3 and accuracy for each label, sorted worst to best."""
    true_labels = val_df["full_label"].tolist()

    label_aps = defaultdict(list)
    label_correct = defaultdict(int)
    label_total = defaultdict(int)

    for true_label, preds in zip(true_labels, predictions):
        ap = average_precision_at_k(true_label, preds, k)
        label_aps[true_label].append(ap)
        label_total[true_label] += 1
        if true_label in preds:
            label_correct[true_label] += 1

    rows = []
    for label in label_aps:
        aps = label_aps[label]
        rows.append({
            "label": label,
            "count": label_total[label],
            "map3": sum(aps) / len(aps),
            "top3_acc": label_correct[label] / label_total[label],
            "top1_acc": sum(1 for ap in aps if ap == 1.0) / len(aps),
        })

    return pd.DataFrame(rows).sort_values("map3", ascending=True)


def sample_failures(val_df, predictions, pred_scores, n=10):
    """Pull out concrete failure examples with the student's actual text."""
    true_labels = val_df["full_label"].tolist()
    failures = []

    for i, (true_label, preds, scores) in enumerate(
        zip(true_labels, predictions, pred_scores)
    ):
        if preds[0] != true_label:
            row = val_df.iloc[i]
            failures.append({
                "idx": i,
                "true_label": true_label,
                "pred_1": preds[0],
                "pred_1_score": scores[0],
                "pred_2": preds[1],
                "pred_2_score": scores[1],
                "pred_3": preds[2],
                "pred_3_score": scores[2],
                "explanation": row["StudentExplanation_clean"],
                "question": row["QuestionText_clean"][:120],
                "answer": row["MC_Answer"],
            })

    # Sample a mix: some high-confidence wrong, some low-confidence wrong
    failures.sort(key=lambda x: x["pred_1_score"], reverse=True)
    high_conf = failures[:n // 2]
    low_conf = failures[-(n // 2):]
    return high_conf + low_conf


def cross_category_errors(val_df, predictions):
    """
    Count how often the model predicts a label from a completely different
    category than the true one (e.g., true=False_Misconception, pred=True_Correct).
    """
    true_labels = val_df["full_label"].tolist()
    true_cats = val_df["Category"].tolist()

    same_cat = 0
    diff_cat = 0
    total_errors = 0

    for true_label, true_cat, preds in zip(true_labels, true_cats, predictions):
        if preds[0] != true_label:
            total_errors += 1
            pred_cat = preds[0].split(":")[0]
            if pred_cat == true_cat:
                same_cat += 1
            else:
                diff_cat += 1

    return total_errors, same_cat, diff_cat


def main():
    print("=" * 60)
    print("Phase 5: Error Analysis (MPNet-base)")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data with same split as training
    df = load_train(DATA_PATH)
    train_df, val_df = train_val_split(df, val_frac=0.2, seed=42)
    print(f"Val set: {len(val_df)} examples, {val_df['full_label'].nunique()} labels\n")

    # Load model and get predictions
    print(f"Loading model from {MODEL_PATH}...")
    model = MisconceptionBiEncoder(model_name=MODEL_PATH)
    predictions, pred_scores, unique_labels = get_predictions(
        model, val_df, train_df, k=3
    )

    # Overall score
    true_labels = val_df["full_label"].tolist()
    overall_map, per_label_map = map_at_k(true_labels, predictions, k=3)
    print(f"\nOverall MAP@3: {overall_map:.4f}")

    top1_correct = sum(
        1 for t, p in zip(true_labels, predictions) if p[0] == t
    )
    print(f"Top-1 accuracy: {top1_correct}/{len(val_df)} ({top1_correct/len(val_df):.1%})")
    top3_correct = sum(
        1 for t, p in zip(true_labels, predictions) if t in p
    )
    print(f"Top-3 accuracy: {top3_correct}/{len(val_df)} ({top3_correct/len(val_df):.1%})")

    # --- 1. Category-level confusion ---
    print("\n" + "=" * 60)
    print("1. Category-Level Confusion Matrix")
    print("=" * 60)

    cat_matrix = category_confusion(val_df, predictions)
    print("\nRows = true category, Columns = predicted category (top-1)")
    print(cat_matrix.to_string())

    cat_matrix.to_csv(OUTPUT_DIR / "category_confusion.csv")
    print(f"\nSaved to {OUTPUT_DIR / 'category_confusion.csv'}")

    # --- 2. Cross-category error rate ---
    print("\n" + "=" * 60)
    print("2. Cross-Category Error Breakdown")
    print("=" * 60)

    total_err, same_cat, diff_cat = cross_category_errors(val_df, predictions)
    print(f"\nTotal errors (top-1 wrong): {total_err}")
    print(f"  Wrong label, same category:      {same_cat} ({same_cat/total_err:.1%})")
    print(f"  Wrong label, different category:  {diff_cat} ({diff_cat/total_err:.1%})")

    # --- 3. Most common confusion pairs ---
    print("\n" + "=" * 60)
    print("3. Most Common Misconception Confusion Pairs")
    print("=" * 60)

    pairs = misconception_confusion_pairs(val_df, predictions, top_n=20)
    print(f"\nTop 20 (true -> predicted) error pairs:")
    pair_rows = []
    for (true_label, pred_label), count in pairs:
        print(f"  {count:4d}x  {true_label}  ->  {pred_label}")
        pair_rows.append({
            "true_label": true_label,
            "pred_label": pred_label,
            "count": count,
        })

    pd.DataFrame(pair_rows).to_csv(OUTPUT_DIR / "confusion_pairs.csv", index=False)

    # --- 4. Per-label breakdown ---
    print("\n" + "=" * 60)
    print("4. Per-Label Performance (worst and best)")
    print("=" * 60)

    label_df = per_label_breakdown(val_df, predictions, k=3)
    label_df.to_csv(OUTPUT_DIR / "per_label_performance.csv", index=False)

    print("\nWorst 10 labels by MAP@3:")
    worst = label_df.head(10)
    for _, row in worst.iterrows():
        print(f"  MAP@3={row['map3']:.3f}  top1={row['top1_acc']:.1%}  "
              f"top3={row['top3_acc']:.1%}  n={int(row['count']):4d}  {row['label']}")

    print("\nBest 10 labels by MAP@3:")
    best = label_df.tail(10)
    for _, row in best.iterrows():
        print(f"  MAP@3={row['map3']:.3f}  top1={row['top1_acc']:.1%}  "
              f"top3={row['top3_acc']:.1%}  n={int(row['count']):4d}  {row['label']}")

    # --- 5. Example failure cases ---
    print("\n" + "=" * 60)
    print("5. Example Failure Cases")
    print("=" * 60)

    failures = sample_failures(val_df, predictions, pred_scores, n=10)

    print("\n--- High-confidence failures (model was sure but wrong) ---")
    for f in failures[:5]:
        print(f"\n  True: {f['true_label']}")
        print(f"  Pred: {f['pred_1']} (score={f['pred_1_score']:.3f})")
        print(f"  Answer: {f['answer']}")
        print(f"  Explanation: {f['explanation'][:150]}")

    print("\n--- Low-confidence failures (model was uncertain) ---")
    for f in failures[5:]:
        print(f"\n  True: {f['true_label']}")
        print(f"  Pred: {f['pred_1']} (score={f['pred_1_score']:.3f})")
        print(f"  Answer: {f['answer']}")
        print(f"  Explanation: {f['explanation'][:150]}")

    failures_df = pd.DataFrame(failures)
    failures_df.to_csv(OUTPUT_DIR / "example_failures.csv", index=False)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("ERROR ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nAll outputs saved to {OUTPUT_DIR}/")
    print("  category_confusion.csv")
    print("  confusion_pairs.csv")
    print("  per_label_performance.csv")
    print("  example_failures.csv")


if __name__ == "__main__":
    main()
