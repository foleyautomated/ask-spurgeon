#!/usr/bin/env python3

import os
import sys
import traceback
from pdf_processor import PDFProcessor

def debug_chunking_step():
    """Debug just the chunking step that's causing the hang"""
    print("=== Debugging Chunking Process ===")
    
    try:
        # Initialize processor
        pdf_dir = "./test_pdfs"
        processor = PDFProcessor(pdf_dir)
        print("PDF processor initialized")
        
        # Get documents (we know this works)
        documents = processor.process_all_pdfs()
        print(f"Successfully loaded {len(documents)} documents")
        
        # Debug each document's chunking individually
        all_chunks = []
        chunk_metadata = []
        
        for i, doc in enumerate(documents):
            print(f"\n--- Processing document {i+1}/{len(documents)}: {doc['filename']} ---")
            print(f"Document length: {len(doc['text'])} characters")
            
            # Show first few characters
            preview = doc['text'][:200].replace('\n', ' ')
            print(f"Text preview: {preview}...")
            
            try:
                print(f"Starting chunking with size=500, overlap=50...")
                
                # Test chunking with debugging
                chunks = processor.chunk_text(
                    doc['text'], 
                    chunk_size=500, 
                    overlap=50
                )
                
                print(f"Successfully created {len(chunks)} chunks")
                
                # Show first chunk preview
                if chunks:
                    first_chunk = chunks[0][:100].replace('\n', ' ')
                    print(f"First chunk preview: {first_chunk}...")
                
                all_chunks.extend(chunks)
                
                # Create metadata for this document's chunks
                for j, chunk in enumerate(chunks):
                    chunk_metadata.append({
                        'filename': doc['filename'],
                        'filepath': doc['filepath'],
                        'chunk_id': j,
                        'total_chunks': len(chunks),
                        'chunk_length': len(chunk),
                        'document_length': doc['length']
                    })
                
                print(f"Added {len(chunks)} chunks to collection")
                
            except Exception as e:
                print(f"ERROR during chunking: {e}")
                traceback.print_exc()
                # Continue with next document
                continue
        
        print(f"\n=== Chunking Summary ===")
        print(f"Total chunks created: {len(all_chunks)}")
        print(f"Total metadata entries: {len(chunk_metadata)}")
        
        # Test a few sample chunks
        if all_chunks:
            print(f"\nSample chunk lengths: {[len(chunk) for chunk in all_chunks[:5]]}")
            
        return all_chunks, chunk_metadata
        
    except Exception as e:
        print(f"ERROR in chunking debug: {e}")
        traceback.print_exc()
        return [], []

if __name__ == "__main__":
    chunks, metadata = debug_chunking_step()
    
    if chunks:
        print(f"\n✓ SUCCESS: Chunking completed with {len(chunks)} total chunks")
    else:
        print(f"\n✗ FAILED: Chunking did not complete")