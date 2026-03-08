"""
Bi-encoder model for student misconception classification.
Wraps sentence-transformers for fine-tuning with contrastive loss.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Union, List, Optional
from sentence_transformers import SentenceTransformer


class MisconceptionBiEncoder:
    """
    Bi-encoder architecture for matching student explanations to misconceptions.
    
    Uses the same encoder for both student inputs and misconception labels,
    trained with contrastive loss to pull matching pairs closer in embedding space.
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None
    ):
        """
        Args:
            model_name: HuggingFace model name or path to fine-tuned checkpoint
            device: 'cuda', 'cpu', or None for auto-detection
        """
        self.model_name = model_name
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # Load the sentence transformer
        self.model = SentenceTransformer(model_name, device=device)
        
    @property
    def embedding_dim(self) -> int:
        """Return the dimensionality of embeddings produced by this model."""
        return self.model.get_sentence_embedding_dimension()
    
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False,
        normalize: bool = True,
        convert_to_numpy: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Encode a list of texts into embeddings.
        
        Args:
            texts: Single text or list of texts to encode
            batch_size: Batch size for encoding
            show_progress: Show progress bar during encoding
            normalize: L2-normalize embeddings (recommended for cosine similarity)
            convert_to_numpy: Return numpy array instead of torch tensor
            
        Returns:
            Array or tensor of shape (n_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]
            
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
            convert_to_numpy=convert_to_numpy,
            device=self.device
        )
        
        return embeddings
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save the model to disk.
        
        Args:
            path: Directory path to save the model
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))
        print(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path], device: Optional[str] = None) -> "MisconceptionBiEncoder":
        """
        Load a fine-tuned model from disk.
        
        Args:
            path: Directory containing the saved model
            device: Device to load model on
            
        Returns:
            MisconceptionBiEncoder instance with loaded weights
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model directory not found: {path}")
        
        return cls(model_name=str(path), device=device)
    
    def get_base_model(self) -> SentenceTransformer:
        """
        Get the underlying SentenceTransformer model.
        Useful for training with sentence-transformers training utilities.
        """
        return self.model


def compute_similarity(
    query_embeddings: np.ndarray,
    corpus_embeddings: np.ndarray
) -> np.ndarray:
    """
    Compute cosine similarity between query and corpus embeddings.
    Assumes embeddings are already L2-normalized.
    
    Args:
        query_embeddings: Shape (n_queries, embedding_dim)
        corpus_embeddings: Shape (n_corpus, embedding_dim)
        
    Returns:
        Similarity matrix of shape (n_queries, n_corpus)
    """
    # Since embeddings are normalized, dot product = cosine similarity
    return query_embeddings @ corpus_embeddings.T


def retrieve_top_k(
    query_embeddings: np.ndarray,
    corpus_embeddings: np.ndarray,
    corpus_labels: List[str],
    k: int = 3
) -> List[List[str]]:
    """
    For each query, retrieve the top-k most similar corpus items.
    
    Args:
        query_embeddings: Shape (n_queries, embedding_dim)
        corpus_embeddings: Shape (n_corpus, embedding_dim)
        corpus_labels: List of labels corresponding to corpus embeddings
        k: Number of top results to return
        
    Returns:
        List of length n_queries, each containing k labels
    """
    similarities = compute_similarity(query_embeddings, corpus_embeddings)
    
    predictions = []
    for query_idx in range(len(query_embeddings)):
        # Get top k indices by similarity
        top_k_indices = np.argsort(similarities[query_idx])[::-1][:k]
        top_k_labels = [corpus_labels[idx] for idx in top_k_indices]
        predictions.append(top_k_labels)
    
    return predictions
