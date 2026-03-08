"""
Interactive misconception predictor.

Loads the fine-tuned MPNet model and cached label embeddings, then lets you
type student explanations and see the top-3 predicted misconceptions.

Usage:
    python demo.py
"""

import json
import warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore", message=".*encountered in matmul.*")

MODEL_DIR = Path("models/ablation_model_mpnet")
MODEL_PATH = MODEL_DIR / "best_model"
LABEL_PATH = MODEL_DIR / "label_list.json"
EMBED_PATH = MODEL_DIR / "label_embeddings.npy"


def load_resources():
    """Load the model, label list, and precomputed label embeddings."""
    from src.model import MisconceptionBiEncoder

    print("Loading model...", end=" ", flush=True)
    model = MisconceptionBiEncoder(model_name=str(MODEL_PATH))
    print("done.")

    with open(LABEL_PATH) as f:
        labels = json.load(f)

    label_embeddings = np.load(EMBED_PATH)

    return model, labels, label_embeddings


def predict(model, labels, label_embeddings, text, k=3):
    """Encode the input text and return the top-k nearest labels with scores."""
    query_emb = model.encode(text, normalize=True)
    scores = query_emb @ label_embeddings.T
    top_idx = np.argsort(scores.squeeze())[::-1][:k]

    results = []
    for idx in top_idx:
        results.append((labels[idx], float(scores.squeeze()[idx])))
    return results


def format_label(label):
    """Make the raw label string more readable."""
    category, misconception = label.split(":", 1)
    if misconception == "NA":
        return f"{category} (no specific misconception)"
    return f"{category}: {misconception}"


def print_banner():
    print()
    print("=" * 58)
    print("  Misconception Predictor — CS273P Final Project")
    print("  Model: MPNet-base (fine-tuned, MAP@3 = 0.81)")
    print("=" * 58)
    print()
    print("Enter a student's math explanation and the model will")
    print("predict the top 3 most likely misconceptions.")
    print()
    print("You can optionally include the answer choice on a")
    print("separate line. Type 'quit' to exit.")
    print()


def get_input():
    """Prompt for explanation and optional answer, return formatted text."""
    print("-" * 58)
    explanation = input("Student explanation: ").strip()
    if not explanation or explanation.lower() in ("quit", "exit", "q"):
        return None

    answer = input("Answer choice (or press Enter to skip): ").strip()

    if answer:
        text = f"Answer: {answer} Explanation: {explanation}"
    else:
        text = explanation

    return text


def print_results(results):
    """Display the top-k predictions."""
    print()
    for rank, (label, score) in enumerate(results, 1):
        bar_len = int(score * 30)
        bar = "#" * bar_len + "." * (30 - bar_len)
        print(f"  {rank}. {format_label(label)}")
        print(f"     confidence: [{bar}] {score:.3f}")
    print()


def main():
    model, labels, label_embeddings = load_resources()
    print_banner()

    while True:
        text = get_input()
        if text is None:
            print("\nGoodbye.")
            break

        results = predict(model, labels, label_embeddings, text, k=3)
        print_results(results)


if __name__ == "__main__":
    main()
