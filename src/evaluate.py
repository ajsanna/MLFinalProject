from collections import defaultdict


def average_precision_at_k(y_true: str, y_pred: list[str], k: int = 3) -> float:
    """AP for a single example with exactly one true label."""
    for rank, pred in enumerate(y_pred[:k], start=1):
        if pred == y_true:
            return 1.0 / rank
    return 0.0


def map_at_k(y_true_list: list[str], y_pred_list: list[list[str]], k: int = 3):
    """
    Returns:
        overall_map: float
        per_label: dict[str, float] — mean AP per true label
    """
    per_label_aps = defaultdict(list)

    for y_true, y_pred in zip(y_true_list, y_pred_list):
        ap = average_precision_at_k(y_true, y_pred, k)
        per_label_aps[y_true].append(ap)

    per_label = {label: sum(aps) / len(aps) for label, aps in per_label_aps.items()}
    all_aps = [ap for aps in per_label_aps.values() for ap in aps]
    overall_map = sum(all_aps) / len(all_aps) if all_aps else 0.0

    return overall_map, per_label
