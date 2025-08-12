#!/usr/bin/env python3

import fitz
import re

def test_regex():
    """Test the regex operation directly"""
    print("=== Testing Regex ===")
    
    # Get some text
    pdf_path = "./test_pdfs/chs1.pdf"
    doc = fitz.open(pdf_path)
    text = doc.load_page(0).get_text()[:1000]  # Just first 1000 chars
    doc.close()
    
    print(f"Original text length: {len(text)}")
    print(f"First 200 chars: {repr(text[:200])}")
    
    # Test the problematic regex
    print("Testing regex substitution...")
    cleaned = re.sub(r'\s+', ' ', text)
    print(f"After regex: {len(cleaned)} chars")
    print(f"First 200 chars: {repr(cleaned[:200])}")
    
    print("SUCCESS: Regex works fine")

if __name__ == "__main__":
    test_regex()