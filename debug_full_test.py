#!/usr/bin/env python3

import os
import sys
from pdf_processor import PDFProcessor
from embeddings import EmbeddingModel
from vector_db import VectorDatabase

def debug_full_processing():
    """Debug the full processing pipeline"""
    print("=== Testing Full PDF Processing ===")
    
    try:
        # Initialize components
        pdf_dir = "./test_pdfs"
        processor = PDFProcessor(pdf_dir)
        print("PDF processor initialized")
        
        # Process all PDFs with debugging
        pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
        print(f"Found {len(pdf_files)} PDF files: {pdf_files}")
        
        documents = []
        
        for i, pdf_file in enumerate(pdf_files):
            print(f"\n--- Processing file {i+1}/{len(pdf_files)}: {pdf_file} ---")
            
            pdf_path = os.path.join(pdf_dir, pdf_file)
            try:
                text = processor.extract_text_from_pdf(pdf_path)
                print(f"  Extracted {len(text)} characters")
                
                if text:
                    documents.append({
                        'filename': pdf_file,
                        'filepath': pdf_path,
                        'text': text,
                        'length': len(text)
                    })
                    print(f"  Added to documents list")
                else:
                    print(f"  No text extracted from {pdf_file}")
                    
            except Exception as e:
                print(f"  Error processing {pdf_file}: {e}")
                continue
        
        print(f"\nSuccessfully processed {len(documents)} documents")
        
        if not documents:
            print("No documents to process further")
            return
        
        # Test chunking
        print("\n=== Testing Chunking ===")
        all_chunks = []
        chunk_metadata = []
        
        for doc in documents:
            print(f"Chunking {doc['filename']}...")
            chunks = processor.chunk_text(doc['text'], chunk_size=500, overlap=50)
            print(f"  Created {len(chunks)} chunks")
            
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
        
        # Test embeddings
        print("\n=== Testing Embeddings ===")
        model = EmbeddingModel('all-MiniLM-L6-v2')
        print("Embedding model loaded")
        
        # Process in smaller batches to avoid memory issues
        batch_size = 10
        embeddings_list = []
        
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}: {len(batch)} chunks")
            
            batch_embeddings = model.embed_texts(batch)
            embeddings_list.append(batch_embeddings)
            print(f"  Generated embeddings: {batch_embeddings.shape}")
        
        # Combine all embeddings
        import numpy as np
        embeddings = np.vstack(embeddings_list)
        print(f"Combined embeddings shape: {embeddings.shape}")
        
        # Test vector database
        print("\n=== Testing Vector Database ===")
        db_path = "./debug_full_vector_db"
        db = VectorDatabase(db_path, embeddings.shape[1])
        
        db.add_documents(embeddings, all_chunks, chunk_metadata)
        print("Documents added to vector database")
        
        db.save()
        print("Vector database saved")
        
        # Check results
        files = os.listdir(db_path)
        print(f"Database files created: {files}")
        
        stats = db.get_stats()
        print("Final stats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print("\n=== SUCCESS: Full processing completed ===")
        
    except Exception as e:
        print(f"Error in full processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_full_processing()