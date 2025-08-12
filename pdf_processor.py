import fitz
import os
from typing import List, Dict
import re


class PDFProcessor:
    def __init__(self, pdf_directory: str):
        self.pdf_directory = pdf_directory
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a single PDF file."""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text()
                
            doc.close()
            
            # Clean up the text
            text = self._clean_text(text)
            return text
            
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
            return ""
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        # First split into lines before normalizing whitespace
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip very short lines that are likely page numbers or artifacts
            if len(line) > 10:
                cleaned_lines.append(line)
        
        # Join lines and normalize whitespace
        text = ' '.join(cleaned_lines)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def process_all_pdfs(self) -> List[Dict[str, str]]:
        """Process all PDFs in the directory and return document metadata."""
        documents = []
        
        if not os.path.exists(self.pdf_directory):
            print(f"PDF directory {self.pdf_directory} does not exist")
            return documents
            
        pdf_files = [f for f in os.listdir(self.pdf_directory) if f.lower().endswith('.pdf')]
        
        print(f"Found {len(pdf_files)} PDF files to process...")
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(self.pdf_directory, pdf_file)
            print(f"Processing: {pdf_file}")
            
            text = self.extract_text_from_pdf(pdf_path)
            
            if text:
                documents.append({
                    'filename': pdf_file,
                    'filepath': pdf_path,
                    'text': text,
                    'length': len(text)
                })
                print(f"  Extracted {len(text)} characters")
            else:
                print(f"  Failed to extract text from {pdf_file}")
                
        print(f"Successfully processed {len(documents)} documents")
        return documents
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks."""
        if len(text) <= chunk_size:
            return [text]
            
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            
            # Try to break at a sentence or word boundary
            if end < len(text):
                # Look for sentence ending
                last_period = text.rfind('.', start, end)
                last_exclamation = text.rfind('!', start, end)
                last_question = text.rfind('?', start, end)
                
                sentence_end = max(last_period, last_exclamation, last_question)
                
                if sentence_end > start:
                    end = sentence_end + 1
                else:
                    # Fall back to word boundary
                    last_space = text.rfind(' ', start, end)
                    if last_space > start:
                        end = last_space
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Calculate new start position
            new_start = end - overlap
            
            # Prevent infinite loop: ensure we always advance
            if new_start <= start:
                new_start = start + 1
            
            start = new_start
            if start >= len(text):
                break
                
        return chunks