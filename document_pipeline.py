from typing import List, Dict, Any
import numpy as np
from pdf_processor import PDFProcessor
from embeddings import EmbeddingModel
from vector_db import VectorDatabase
import config


class DocumentPipeline:
    def __init__(self, pdf_dir: str = None, vector_db_path: str = None):
        """Initialize the document processing pipeline."""
        self.pdf_dir = pdf_dir or config.PDF_DIR
        self.vector_db_path = vector_db_path or config.VECTOR_DB_PATH
        
        # Initialize components
        self.pdf_processor = PDFProcessor(self.pdf_dir)
        self.embedding_model = EmbeddingModel(config.EMBEDDING_MODEL)
        self.vector_db = VectorDatabase(
            self.vector_db_path, 
            self.embedding_model.get_embedding_dimension()
        )
        
    def process_and_index_documents(self, chunk_size: int = None, chunk_overlap: int = None):
        """Process all PDFs and create/update the vector index."""
        chunk_size = chunk_size or config.CHUNK_SIZE
        chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP
        
        print("Starting document processing pipeline...")
        print(f"PDF Directory: {self.pdf_dir}")
        print(f"Vector DB Path: {self.vector_db_path}")
        print(f"Chunk Size: {chunk_size}, Overlap: {chunk_overlap}")
        print("-" * 50)
        
        # Step 1: Extract text from all PDFs
        documents = self.pdf_processor.process_all_pdfs()
        
        if not documents:
            print("No documents found to process")
            return
        
        print(f"\nSuccessfully extracted text from {len(documents)} documents")
        
        # Step 2: Chunk all documents
        all_chunks = []
        chunk_metadata = []
        
        print("\nChunking documents...")
        for doc in documents:
            chunks = self.pdf_processor.chunk_text(
                doc['text'], 
                chunk_size=chunk_size, 
                overlap=chunk_overlap
            )
            
            print(f"  {doc['filename']}: {len(chunks)} chunks")
            
            # Create metadata for each chunk
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_metadata.append({
                    'filename': doc['filename'],
                    'filepath': doc['filepath'],
                    'chunk_id': i,
                    'total_chunks': len(chunks),
                    'chunk_length': len(chunk),
                    'document_length': doc['length']
                })
        
        print(f"\nTotal chunks created: {len(all_chunks)}")
        
        # Step 3: Generate embeddings
        print("\nGenerating embeddings...")
        embeddings = self.embedding_model.embed_texts(all_chunks, batch_size=8)
        
        print(f"Generated embeddings shape: {embeddings.shape}")
        
        # Step 4: Add to vector database
        print("\nAdding to vector database...")
        self.vector_db.add_documents(embeddings, all_chunks, chunk_metadata)
        
        # Step 5: Save database
        print("\nSaving vector database...")
        self.vector_db.save()
        
        print("\n" + "="*50)
        print("Document processing completed successfully!")
        
        # Print final statistics
        stats = self.vector_db.get_stats()
        print(f"Final Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    def load_existing_database(self) -> bool:
        """Load an existing vector database."""
        print("Loading existing vector database...")
        success = self.vector_db.load()
        
        if success:
            stats = self.vector_db.get_stats()
            print("Database loaded successfully!")
            for key, value in stats.items():
                print(f"  {key}: {value}")
        else:
            print("Failed to load existing database")
            
        return success
    
    def search_documents(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant document chunks."""
        if self.vector_db.index is None or self.vector_db.index.ntotal == 0:
            print("No documents in database. Please process documents first.")
            return []
        
        print(f"Searching for: '{query}'")
        
        # Generate query embedding
        query_embedding = self.embedding_model.embed_query(query)
        
        # Search vector database
        results = self.vector_db.search(query_embedding, k=num_results)
        
        print(f"Found {len(results)} relevant chunks")
        
        return results
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get information about the current database."""
        if self.vector_db.index is None:
            # Try to load existing database
            if not self.load_existing_database():
                return {"status": "No database available"}
        
        return self.vector_db.get_stats()
    
    def preview_chunks(self, filename: str = None, max_chunks: int = 3) -> List[Dict[str, Any]]:
        """Preview chunks from the database."""
        if not self.vector_db.chunks:
            if not self.load_existing_database():
                return []
        
        preview = []
        for i, (chunk, metadata) in enumerate(zip(self.vector_db.chunks, self.vector_db.documents)):
            if filename and metadata['filename'] != filename:
                continue
                
            preview.append({
                'index': i,
                'filename': metadata['filename'],
                'chunk_id': metadata['chunk_id'],
                'chunk_preview': chunk[:200] + "..." if len(chunk) > 200 else chunk,
                'chunk_length': len(chunk)
            })
            
            if len(preview) >= max_chunks:
                break
                
        return preview