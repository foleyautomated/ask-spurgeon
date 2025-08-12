import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Tuple, Any
import json


class VectorDatabase:
    def __init__(self, db_path: str, embedding_dimension: int = 384):
        """Initialize the vector database.
        
        Args:
            db_path: Path to store the database files
            embedding_dimension: Dimension of the embedding vectors
        """
        self.db_path = db_path
        self.embedding_dimension = embedding_dimension
        self.index = None
        self.documents = []  # Store document metadata
        self.chunks = []     # Store text chunks
        
        # Create directory if it doesn't exist
        os.makedirs(db_path, exist_ok=True)
        
        # File paths
        self.index_path = os.path.join(db_path, "faiss.index")
        self.metadata_path = os.path.join(db_path, "metadata.json")
        self.chunks_path = os.path.join(db_path, "chunks.pkl")
        
    def create_index(self):
        """Create a new FAISS index."""
        print(f"Creating new FAISS index with dimension {self.embedding_dimension}")
        
        # Use IndexFlatIP for inner product (cosine similarity with normalized vectors)
        self.index = faiss.IndexFlatIP(self.embedding_dimension)
        
        # Try to use GPU if available
        try:
            if faiss.get_num_gpus() > 0:
                print(f"GPU detected, moving index to GPU")
                gpu_resources = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(gpu_resources, 0, self.index)
                print(f"Index moved to GPU successfully")
            else:
                print(f"No GPU detected, using CPU")
        except Exception as e:
            print(f"Could not use GPU, falling back to CPU: {e}")
        
        print(f"Index created successfully")
        
    def add_documents(self, embeddings: np.ndarray, chunks: List[str], 
                     document_metadata: List[Dict[str, Any]]):
        """Add documents to the vector database.
        
        Args:
            embeddings: Numpy array of embeddings (n_chunks, embedding_dim)
            chunks: List of text chunks
            document_metadata: List of metadata for each chunk
        """
        if self.index is None:
            self.create_index()
            
        # Ensure embeddings are float32
        embeddings = embeddings.astype('float32')
        
        print(f"Adding {len(embeddings)} embeddings to the index")
        
        # Add embeddings to FAISS index
        self.index.add(embeddings)
        
        # Store text chunks and metadata
        self.chunks.extend(chunks)
        self.documents.extend(document_metadata)
        
        print(f"Index now contains {self.index.ntotal} vectors")
        
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar chunks.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of dictionaries containing chunk text, metadata, and similarity scores
        """
        if self.index is None or self.index.ntotal == 0:
            print("No documents in the index")
            return []
            
        # Ensure query embedding is the right shape and type
        query_embedding = query_embedding.astype('float32')
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        # Search the index
        scores, indices = self.index.search(query_embedding, min(k, self.index.ntotal))
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.chunks):  # Ensure index is valid
                result = {
                    'chunk': self.chunks[idx],
                    'metadata': self.documents[idx],
                    'similarity_score': float(score),
                    'rank': i + 1
                }
                results.append(result)
                
        return results
    
    def save(self):
        """Save the index and metadata to disk."""
        if self.index is None:
            print("No index to save")
            return
            
        print(f"Saving index to {self.db_path}")
        
        # Save FAISS index (convert GPU index to CPU for saving)
        try:
            if hasattr(self.index, 'index'):  # GPU index
                print("Converting GPU index to CPU for saving")
                cpu_index = faiss.index_gpu_to_cpu(self.index)
                faiss.write_index(cpu_index, self.index_path)
            else:  # CPU index
                faiss.write_index(self.index, self.index_path)
        except Exception as e:
            print(f"Error saving index, trying direct save: {e}")
            faiss.write_index(self.index, self.index_path)
        
        # Save metadata
        metadata = {
            'total_documents': len(set(doc['filename'] for doc in self.documents)),
            'total_chunks': len(self.chunks),
            'embedding_dimension': self.embedding_dimension
        }
        
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        # Save chunks and document metadata
        data = {
            'chunks': self.chunks,
            'documents': self.documents
        }
        
        with open(self.chunks_path, 'wb') as f:
            pickle.dump(data, f)
            
        print(f"Database saved successfully")
        print(f"  - {metadata['total_documents']} documents")
        print(f"  - {metadata['total_chunks']} chunks")
        
    def load(self) -> bool:
        """Load the index and metadata from disk.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        if not all(os.path.exists(path) for path in [self.index_path, self.metadata_path, self.chunks_path]):
            print("Database files not found")
            return False
            
        try:
            print(f"Loading index from {self.db_path}")
            
            # Load FAISS index
            self.index = faiss.read_index(self.index_path)
            
            # Move to GPU if available
            try:
                if faiss.get_num_gpus() > 0:
                    print("GPU detected, moving loaded index to GPU")
                    gpu_resources = faiss.StandardGpuResources()
                    self.index = faiss.index_cpu_to_gpu(gpu_resources, 0, self.index)
                    print("Index moved to GPU successfully")
                else:
                    print("No GPU detected, keeping index on CPU")
            except Exception as e:
                print(f"Could not move index to GPU, keeping on CPU: {e}")
            
            # Load metadata
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)
                
            # Load chunks and document metadata
            with open(self.chunks_path, 'rb') as f:
                data = pickle.load(f)
                
            self.chunks = data['chunks']
            self.documents = data['documents']
            
            print(f"Database loaded successfully")
            print(f"  - {metadata['total_documents']} documents")
            print(f"  - {metadata['total_chunks']} chunks")
            print(f"  - {self.index.ntotal} vectors in index")
            
            return True
            
        except Exception as e:
            print(f"Error loading database: {str(e)}")
            return False
            
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        if self.index is None:
            return {"status": "No index loaded"}
            
        unique_docs = set(doc['filename'] for doc in self.documents)
        
        return {
            "total_documents": len(unique_docs),
            "total_chunks": len(self.chunks),
            "vectors_in_index": self.index.ntotal,
            "embedding_dimension": self.embedding_dimension,
            "database_path": self.db_path
        }