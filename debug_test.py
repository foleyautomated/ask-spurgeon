#!/usr/bin/env python3

import os
import fitz
import sys
from embeddings import EmbeddingModel
from vector_db import VectorDatabase
import time

def test_pdf_processing():
    """Test PDF processing step by step"""
    print("=== Testing PDF Processing ===")
    
    pdf_dir = "./test_pdfs"
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
    print(f"Found {len(pdf_files)} PDF files")
    
    # Test processing just the first PDF
    if pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_files[0])
        print(f"Testing with: {pdf_files[0]}")
        
        try:
            doc = fitz.open(pdf_path)
            print(f"Opened PDF with {len(doc)} pages")
            
            text = ""
            for page_num in range(min(2, len(doc))):  # Only process first 2 pages
                print(f"Processing page {page_num + 1}")
                page = doc.load_page(page_num)
                page_text = page.get_text()
                text += page_text
                print(f"  Extracted {len(page_text)} characters")
                
            doc.close()
            print(f"Total extracted: {len(text)} characters")
            
            return text
            
        except Exception as e:
            print(f"Error processing PDF: {e}")
            return None
    
    return None

def test_embeddings(text):
    """Test embedding generation"""
    print("\n=== Testing Embeddings ===")
    
    try:
        model = EmbeddingModel('all-MiniLM-L6-v2')
        print("Embedding model loaded")
        
        # Test with small text chunk
        test_chunks = [text[:500], text[500:1000]] if len(text) > 1000 else [text[:500]]
        print(f"Testing with {len(test_chunks)} chunks")
        
        embeddings = model.embed_texts(test_chunks)
        print(f"Generated embeddings shape: {embeddings.shape}")
        
        return embeddings, test_chunks
        
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None, None

def test_vector_db(embeddings, chunks):
    """Test vector database creation"""
    print("\n=== Testing Vector Database ===")
    
    try:
        db_path = "./debug_vector_db"
        db = VectorDatabase(db_path, embeddings.shape[1])
        print("Vector database initialized")
        
        metadata = [{"filename": "test.pdf", "chunk_id": i} for i in range(len(chunks))]
        
        db.add_documents(embeddings, chunks, metadata)
        print("Documents added to database")
        
        db.save()
        print("Database saved")
        
        # Check if files were created
        files = os.listdir(db_path)
        print(f"Created files: {files}")
        
        return True
        
    except Exception as e:
        print(f"Error with vector database: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        # Step 1: Test PDF processing
        text = test_pdf_processing()
        if not text:
            print("PDF processing failed")
            sys.exit(1)
            
        # Step 2: Test embeddings
        embeddings, chunks = test_embeddings(text)
        if embeddings is None:
            print("Embedding generation failed")
            sys.exit(1)
            
        # Step 3: Test vector database
        success = test_vector_db(embeddings, chunks)
        if success:
            print("\n=== SUCCESS: All components working ===")
        else:
            print("\n=== FAILED: Vector database issue ===")
            
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()