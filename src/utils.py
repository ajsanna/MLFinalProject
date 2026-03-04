import numpy as np
from sentence_transformers import SentenceTransformer


def encode_texts(
    texts: list[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64,
    show_progress: bool = True,
) -> np.ndarray:
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings


def cosine_similarity_matrix(queries: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """
    Since embeddings are L2-normalized, dot product = cosine similarity.
    Returns shape (len(queries), len(corpus)).
    """
    return queries @ corpus.T
