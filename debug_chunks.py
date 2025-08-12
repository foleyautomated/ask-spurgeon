#!/usr/bin/env python3

import os
import sys
from pdf_processor import PDFProcessor

def debug_chunking():
    """Debug the chunking process"""
    print("=== Testing Chunking Process ===")
    
    try:
        pdf_dir = "./test_pdfs"
        processor = PDFProcessor(pdf_dir)
        
        # Get all documents
        documents = processor.process_all_pdfs()
        print(f"Loaded {len(documents)} documents")
        
        # Calculate total chunks
        all_chunks = []
        chunk_metadata = []
        
        total_chars = 0
        for doc in documents:
            total_chars += len(doc['text'])
            print(f"{doc['filename']}: {len(doc['text'])} characters")
            
            chunks = processor.chunk_text(doc['text'], chunk_size=500, overlap=50)
            print(f"  -> {len(chunks)} chunks")
            
            all_chunks.extend(chunks)
            
            # Add metadata
            for i, chunk in enumerate(chunks):
                chunk_metadata.append({
                    'filename': doc['filename'],
                    'filepath': doc['filepath'],
                    'chunk_id': i,
                    'total_chunks': len(chunks),
                    'chunk_length': len(chunk),
                    'document_length': doc['length']
                })
        
        print(f"\nTotal characters: {total_chars}")
        print(f"Total chunks: {len(all_chunks)}")
        
        # Estimate memory usage
        avg_chunk_size = total_chars / len(all_chunks) if all_chunks else 0
        print(f"Average chunk size: {avg_chunk_size:.0f} characters")
        
        # Test embedding with different batch sizes
        print(f"\n=== Testing Embedding Generation ===")
        
        from embeddings import EmbeddingModel
        model = EmbeddingModel('all-MiniLM-L6-v2')
        
        # Try small batch first
        test_chunks = all_chunks[:5]  # Just first 5 chunks
        print(f"Testing with {len(test_chunks)} chunks...")
        
        embeddings = model.embed_texts(test_chunks, batch_size=2)
        print(f"Success! Generated embeddings: {embeddings.shape}")
        
        # Now try all chunks with smaller batch size
        print(f"\nTesting all {len(all_chunks)} chunks with batch_size=8...")
        all_embeddings = model.embed_texts(all_chunks, batch_size=8)
        print(f"Success! Generated all embeddings: {all_embeddings.shape}")
        
        print(f"\n=== SUCCESS: Chunking and embedding works ===")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_chunking()