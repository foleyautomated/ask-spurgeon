from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List
import os


class EmbeddingModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the embedding model.
        
        all-MiniLM-L6-v2 is a good balance of performance and quality:
        - 384 dimensions
        - Fast inference
        - Good semantic understanding
        """
        self.model_name = model_name
        self.model = None
        self._load_model()
        
    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(f"Model loaded successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"Error loading model {self.model_name}: {str(e)}")
            raise
    
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        if not self.model:
            raise ValueError("Model not loaded")
            
        if not texts:
            return np.array([])
            
        print(f"Generating embeddings for {len(texts)} texts...")
        
        try:
            # Generate embeddings in batches for efficiency
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True  # Normalize for cosine similarity
            )
            
            print(f"Generated embeddings with shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            print(f"Error generating embeddings: {str(e)}")
            raise
    
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a single query."""
        if not self.model:
            raise ValueError("Model not loaded")
            
        try:
            embedding = self.model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            return embedding[0]  # Return single embedding vector
            
        except Exception as e:
            print(f"Error generating query embedding: {str(e)}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors."""
        if not self.model:
            raise ValueError("Model not loaded")
        return self.model.get_sentence_embedding_dimension()
    
    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))