import pandas as pd
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict
from collections import Counter
import numpy as np


def load_train(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def train_val_split(df: pd.DataFrame, val_frac: float = 0.2, seed: int = 42):
    label_counts = df["full_label"].value_counts()
    rare_labels = label_counts[label_counts < 2].index
    rare_mask = df["full_label"].isin(rare_labels)

    rare_df = df[rare_mask]
    common_df = df[~rare_mask]

    train_common, val_common = train_test_split(
        common_df,
        test_size=val_frac,
        stratify=common_df["full_label"],
        random_state=seed,
    )

    train_df = pd.concat([train_common, rare_df], ignore_index=True)
    val_df = val_common.reset_index(drop=True)

    return train_df, val_df


def get_label_set(df: pd.DataFrame) -> list[str]:
    return sorted(df["full_label"].unique().tolist())


def create_contrastive_pairs(
    df: pd.DataFrame,
    input_col: str = "model_input",
    label_col: str = "full_label"
) -> Tuple[List[str], List[str]]:
    """
    Create (input, label) pairs for contrastive learning.
    
    For MultipleNegativesRankingLoss, we need pairs of (anchor, positive).
    The loss automatically samples in-batch negatives.
    
    Args:
        df: DataFrame with student responses
        input_col: Column containing the student input text
        label_col: Column containing the label
        
    Returns:
        Tuple of (input_texts, labels)
    """
    input_texts = df[input_col].tolist()
    labels = df[label_col].tolist()
    
    return input_texts, labels


def create_label_corpus(df: pd.DataFrame, label_col: str = "full_label") -> Dict[str, str]:
    """
    Create a corpus of unique labels for encoding.
    
    For bi-encoder retrieval, we encode each unique label once and use it as
    a prototype. At inference time, we find the nearest label prototypes.
    
    Args:
        df: DataFrame with training data
        label_col: Column containing labels
        
    Returns:
        Dictionary mapping label -> label text (for now, just the label itself)
    """
    unique_labels = df[label_col].unique()
    
    # For now, use the label string itself as the text to encode
    # In a more sophisticated approach, you could create descriptive text
    # for each label (e.g., "The student believes that...")
    label_corpus = {label: label for label in unique_labels}
    
    return label_corpus


def oversample_rare_labels(
    df: pd.DataFrame,
    label_col: str = "full_label",
    target_min_count: int = 10,
    seed: int = 42
) -> pd.DataFrame:
    """
    Oversample rare labels to ensure each label has at least target_min_count examples.
    This helps with the severe class imbalance in the dataset.
    
    Args:
        df: Training DataFrame
        label_col: Column containing labels
        target_min_count: Minimum number of examples per label
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with rare labels oversampled
    """
    np.random.seed(seed)
    
    label_counts = df[label_col].value_counts()
    rare_labels = label_counts[label_counts < target_min_count].index
    
    dfs_to_concat = [df]
    
    for label in rare_labels:
        label_df = df[df[label_col] == label]
        current_count = len(label_df)
        needed = target_min_count - current_count
        
        if needed > 0:
            # Sample with replacement to reach target count
            oversampled = label_df.sample(n=needed, replace=True, random_state=seed)
            dfs_to_concat.append(oversampled)
    
    result_df = pd.concat(dfs_to_concat, ignore_index=True)
    
    # Shuffle the result
    result_df = result_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    
    return result_df


def get_label_weights(df: pd.DataFrame, label_col: str = "full_label") -> Dict[str, float]:
    """
    Calculate inverse frequency weights for labels to handle class imbalance.
    
    Args:
        df: Training DataFrame
        label_col: Column containing labels
        
    Returns:
        Dictionary mapping label -> weight
    """
    label_counts = df[label_col].value_counts()
    total = len(df)
    
    # Inverse frequency weighting
    weights = {label: total / count for label, count in label_counts.items()}
    
    # Normalize so average weight is 1.0
    avg_weight = sum(weights.values()) / len(weights)
    weights = {label: w / avg_weight for label, w in weights.items()}
    
    return weights
