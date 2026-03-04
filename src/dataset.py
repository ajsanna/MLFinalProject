import pandas as pd
from sklearn.model_selection import train_test_split


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
