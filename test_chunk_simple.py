#!/usr/bin/env python3

def test_chunking_logic():
    """Test the chunking logic with simple text"""
    print("=== Testing Chunking Logic ===")
    
    # Simple test text
    test_text = "This is a test sentence. Here is another sentence! And a question? More text follows. Final sentence here."
    print(f"Test text length: {len(test_text)}")
    print(f"Test text: {test_text}")
    
    # Test chunking parameters
    chunk_size = 50
    overlap = 10
    
    print(f"\nChunking with size={chunk_size}, overlap={overlap}")
    
    # Manual implementation to debug
    chunks = []
    start = 0
    iteration = 0
    
    while start < len(test_text):
        iteration += 1
        print(f"\nIteration {iteration}:")
        print(f"  start={start}, text_len={len(test_text)}")
        
        if iteration > 20:  # Safety break
            print("  SAFETY BREAK - too many iterations!")
            break
        
        end = min(start + chunk_size, len(test_text))
        print(f"  initial end={end}")
        
        # Try to break at sentence or word boundary
        if end < len(test_text):
            # Look for sentence ending
            last_period = test_text.rfind('.', start, end)
            last_exclamation = test_text.rfind('!', start, end)
            last_question = test_text.rfind('?', start, end)
            
            sentence_end = max(last_period, last_exclamation, last_question)
            print(f"  sentence markers at: {last_period}, {last_exclamation}, {last_question}")
            print(f"  best sentence_end={sentence_end}")
            
            if sentence_end > start:
                end = sentence_end + 1
                print(f"  adjusted end to sentence boundary: {end}")
            else:
                # Fall back to word boundary
                last_space = test_text.rfind(' ', start, end)
                print(f"  no sentence boundary, last_space={last_space}")
                if last_space > start:
                    end = last_space
                    print(f"  adjusted end to word boundary: {end}")
        
        chunk = test_text[start:end].strip()
        print(f"  chunk: '{chunk}' (length: {len(chunk)})")
        
        if chunk:
            chunks.append(chunk)
        
        new_start = end - overlap
        print(f"  new_start would be: {new_start}")
        
        if new_start <= start:
            print(f"  WARNING: new_start ({new_start}) <= current start ({start})")
            print(f"  This could cause infinite loop!")
            # Force advancement
            new_start = start + 1
            print(f"  Forcing advancement to: {new_start}")
        
        start = new_start
        
        if start >= len(test_text):
            print(f"  start ({start}) >= text_len ({len(test_text)}), breaking")
            break
    
    print(f"\nFinal result: {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}: '{chunk}'")

if __name__ == "__main__":
    test_chunking_logic()