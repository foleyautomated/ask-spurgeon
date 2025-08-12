#!/usr/bin/env python3

import os
import sys
from pdf_processor import PDFProcessor

def debug_pdf_processing():
    """Debug just the PDF processing"""
    print("=== Testing PDF Processing Only ===")
    
    try:
        pdf_dir = "./test_pdfs"
        processor = PDFProcessor(pdf_dir)
        print("PDF processor initialized")
        
        # List files
        pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
        print(f"Found {len(pdf_files)} PDF files: {pdf_files}")
        
        # Process each file individually
        for i, pdf_file in enumerate(pdf_files):
            print(f"\n--- Processing {i+1}/{len(pdf_files)}: {pdf_file} ---")
            
            pdf_path = os.path.join(pdf_dir, pdf_file)
            print(f"Full path: {pdf_path}")
            
            # Check if file exists and is readable
            if not os.path.exists(pdf_path):
                print(f"  ERROR: File does not exist")
                continue
                
            file_size = os.path.getsize(pdf_path)
            print(f"  File size: {file_size} bytes")
            
            try:
                print(f"  Attempting to extract text...")
                text = processor.extract_text_from_pdf(pdf_path)
                print(f"  SUCCESS: Extracted {len(text)} characters")
                
                # Show first 100 chars
                preview = text[:100].replace('\n', ' ')
                print(f"  Preview: {preview}...")
                
            except Exception as e:
                print(f"  ERROR extracting from {pdf_file}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"\n=== Completed individual file processing ===")
        
        # Now test the process_all_pdfs method
        print(f"\n=== Testing process_all_pdfs method ===")
        documents = processor.process_all_pdfs()
        print(f"process_all_pdfs returned {len(documents)} documents")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_pdf_processing()