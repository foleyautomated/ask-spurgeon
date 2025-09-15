from openai import OpenAI
from typing import List, Dict, Any
import config
import tiktoken


class LLMClient:
    def __init__(self, api_key: str = None, model: str = None):
        """Initialize the OpenAI client."""
        self.api_key = api_key or config.OPENAI_API_KEY
        self.model = model or config.OPENAI_MODEL
        
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Please set OPENAI_API_KEY in your .env file")
        
        self.client = OpenAI(api_key=self.api_key)
        
        # Initialize tokenizer for the model
        try:
            self.tokenizer = tiktoken.encoding_for_model(self.model)
        except KeyError:
            # Fallback for newer models
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string."""
        return len(self.tokenizer.encode(text))
    
    def create_rag_prompt(self, query: str, relevant_chunks: List[Dict[str, Any]], 
                         max_context_tokens: int = 3000) -> str:
        """Create a RAG prompt with query and relevant context."""
        
        # Base prompt template
        system_prompt = """
You are Charles Spurgeon. 
Instructions:
1. Use ONLY the information provided in the context to answer the question
2. If the context doesn't contain enough information to answer the question, demand that the user stay focused on theological matters. 
3. Cite the relevant document names when possible
5. If asked about something not in the context, explain that you can only answer based on the provided documents
6. IMPORTANT: Speak in the first person, as if you were Charles Spurgeon--using a passionate, 19th century preacher's diction. Try to give decisive answers. 

Context from documents:
"""

        # Add context chunks
        context_parts = []
        total_tokens = self.count_tokens(system_prompt + query)
        
        for i, chunk_data in enumerate(relevant_chunks):
            chunk_text = chunk_data['chunk']
            filename = chunk_data['metadata']['filename']
            
            # Format chunk with source info
            chunk_context = f"\n--- From {filename} ---\n{chunk_text}\n"
            chunk_tokens = self.count_tokens(chunk_context)
            
            # Check if adding this chunk would exceed token limit
            if total_tokens + chunk_tokens > max_context_tokens:
                break
                
            context_parts.append(chunk_context)
            total_tokens += chunk_tokens
        
        # Combine all context
        full_context = system_prompt + "".join(context_parts)
        
        # Add user question
        full_prompt = f"{full_context}\n\nQuestion: {query}\n\nAnswer:"
        
        return full_prompt
    
    def generate_response(self, query: str, relevant_chunks: List[Dict[str, Any]], 
                         max_context_tokens: int = 3000) -> Dict[str, Any]:
        """Generate a response using RAG approach."""
        
        # Create the prompt
        prompt = self.create_rag_prompt(query, relevant_chunks, max_context_tokens)
        
        # Count tokens for cost estimation
        prompt_tokens = self.count_tokens(prompt)
        
        try:
            # Generate response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            
            # Extract response text
            answer = response.choices[0].message.content.strip()
            
            # Calculate token usage
            completion_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens
            
            return {
                "answer": answer,
                "sources": [chunk['metadata']['filename'] for chunk in relevant_chunks],
                "chunks_used": len(relevant_chunks),
                "tokens": {
                    "prompt": prompt_tokens,
                    "completion": completion_tokens,
                    "total": total_tokens
                },
                "model": self.model,
                "success": True
            }
            
        except Exception as e:
            return {
                "error": f"Error generating response: {str(e)}",
                "success": False
            }
    
    def generate_simple_response(self, prompt: str) -> Dict[str, Any]:
        """Generate a simple response without RAG context."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content.strip()
            
            return {
                "answer": answer,
                "tokens": {
                    "prompt": response.usage.prompt_tokens,
                    "completion": response.usage.completion_tokens,
                    "total": response.usage.total_tokens
                },
                "model": self.model,
                "success": True
            }
            
        except Exception as e:
            return {
                "error": f"Error generating response: {str(e)}",
                "success": False
            }