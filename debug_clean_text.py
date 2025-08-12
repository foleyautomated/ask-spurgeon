#!/usr/bin/env python3

import fitz
from pdf_processor import PDFProcessor

def test_clean_text():
    """Test if the _clean_text method is causing issues"""
    print("=== Testing Text Cleaning ===")
    
    # Extract raw text from one PDF
    pdf_path = "./test_pdfs/chs1.pdf"
    print(f"Testing with: {pdf_path}")
    
    try:
        # Get raw text from PDF
        doc = fitz.open(pdf_path)
        raw_text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            raw_text += page.get_text()
        doc.close()
        
        print(f"Raw text length: {len(raw_text)} characters")
        
        # Test the cleaning method
        processor = PDFProcessor("./test_pdfs")
        print("Testing _clean_text method...")
        
        cleaned_text = processor._clean_text(raw_text)
        print(f"Cleaned text length: {len(cleaned_text)} characters")
        
        # Test chunking with cleaned text
        print("Testing chunking...")
        chunks = processor.chunk_text(cleaned_text[:5000])  # Just first 5000 chars
        print(f"Created {len(chunks)} chunks from sample")
        
        print("SUCCESS: Text processing works")
        return cleaned_text
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_clean_text()