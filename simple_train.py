#!/usr/bin/env python3

import os
from pdf_processor import PDFProcessor
from embeddings import EmbeddingModel
from vector_db import VectorDatabase

def simple_train():
    """Simplified training without CLI overhead"""
    print("=== Simple Training Process ===")
    
    pdf_dir = "./test_pdfs"
    db_path = "./simple_vector_db"
    
    print(f"PDF Directory: {pdf_dir}")
    print(f"Vector DB Path: {db_path}")
    
    # Step 1: Process PDFs
    print("\n1. Processing PDFs...")
    processor = PDFProcessor(pdf_dir)
    documents = processor.process_all_pdfs()
    
    if not documents:
        print("No documents found!")
        return False
    
    print(f"Processed {len(documents)} documents")
    
    # Step 2: Create chunks
    print("\n2. Creating chunks...")
    all_chunks = []
    chunk_metadata = []
    
    for doc in documents:
        chunks = processor.chunk_text(doc['text'], chunk_size=500, overlap=50)
        print(f"  {doc['filename']}: {len(chunks)} chunks")
        
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
    
    print(f"Total chunks: {len(all_chunks)}")
    
    # Step 3: Generate embeddings (with very small batch size)
    print("\n3. Generating embeddings...")
    model = EmbeddingModel('all-MiniLM-L6-v2')
    
    # Use very small batch size to avoid hanging
    embeddings = model.embed_texts(all_chunks, batch_size=2)
    print(f"Generated embeddings: {embeddings.shape}")
    
    # Step 4: Create vector database
    print("\n4. Creating vector database...")
    db = VectorDatabase(db_path, embeddings.shape[1])
    db.add_documents(embeddings, all_chunks, chunk_metadata)
    
    # Step 5: Save database
    print("\n5. Saving database...")
    db.save()
    
    # Verify files were created
    db_files = os.listdir(db_path)
    print(f"Database files: {db_files}")
    
    print("\n=== SUCCESS: Training completed ===")
    return True

if __name__ == "__main__":
    try:
        success = simple_train()
        if success:
            print("Training successful!")
        else:
            print("Training failed!")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()