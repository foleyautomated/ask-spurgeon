#!/usr/bin/env python3

import click
import sys
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich import print as rprint
from document_pipeline import DocumentPipeline
from llm_client import LLMClient
import config

console = Console()


class RAGChatbot:
    def __init__(self, vector_db_path: str = None, num_results: int = 5):
        """Initialize the RAG chatbot."""
        self.vector_db_path = vector_db_path or config.VECTOR_DB_PATH
        self.num_results = num_results
        
        # Initialize components
        console.print("[yellow]Initializing chatbot components...[/yellow]")
        
        try:
            self.pipeline = DocumentPipeline(vector_db_path=self.vector_db_path)
            self.llm_client = LLMClient()
            
            # Load existing database
            if not self.pipeline.load_existing_database():
                console.print("[red]❌ Could not load vector database[/red]")
                console.print("[yellow]Please run training first: python train.py[/yellow]")
                sys.exit(1)
                
            console.print("[green]✅ Chatbot initialized successfully![/green]")
            
        except ValueError as e:
            console.print(f"[red]❌ Configuration error: {str(e)}[/red]")
            console.print("[yellow]Please check your .env file and ensure OPENAI_API_KEY is set[/yellow]")
            sys.exit(1)
        except Exception as e:
            console.print(f"[red]❌ Initialization error: {str(e)}[/red]")
            sys.exit(1)
    
    def search_and_respond(self, query: str) -> dict:
        """Search for relevant chunks and generate response."""
        console.print(f"[blue]🔍 Searching for relevant information...[/blue]")
        
        # Search for relevant chunks
        relevant_chunks = self.pipeline.search_documents(query, self.num_results)
        
        if not relevant_chunks:
            return {
                "answer": "I couldn't find any relevant information in the documents to answer your question.",
                "sources": [],
                "chunks_used": 0,
                "success": True
            }
        
        console.print(f"[green]📄 Found {len(relevant_chunks)} relevant chunks[/green]")
        
        # Generate response using LLM
        console.print(f"[blue]🤖 Generating response...[/blue]")
        response = self.llm_client.generate_response(query, relevant_chunks)
        
        return response
    
    def display_response(self, query: str, response: dict):
        """Display the chatbot response in a nice format."""
        if not response["success"]:
            console.print(f"[red]❌ {response['error']}[/red]")
            return
        
        # Display the answer
        console.print(Panel(
            Markdown(response["answer"]),
            title="[bold green]🤖 Answer[/bold green]",
            border_style="green"
        ))
        
        # Display sources and metadata
        if response.get("sources"):
            sources_table = Table(title="📚 Sources", border_style="blue")
            sources_table.add_column("Document", style="cyan")
            sources_table.add_column("Chunks Used", style="green")
            
            unique_sources = list(set(response["sources"]))
            for source in unique_sources:
                chunk_count = response["sources"].count(source)
                sources_table.add_row(source, str(chunk_count))
            
            console.print(sources_table)
        
        # Display token usage
        if response.get("tokens"):
            tokens = response["tokens"]
            console.print(f"[dim]📊 Token usage: {tokens['total']} total "
                         f"({tokens['prompt']} prompt + {tokens['completion']} completion)[/dim]")
    
    def display_chunk_details(self, query: str):
        """Display detailed information about retrieved chunks."""
        relevant_chunks = self.pipeline.search_documents(query, self.num_results)
        
        if not relevant_chunks:
            console.print("[yellow]No relevant chunks found[/yellow]")
            return
        
        console.print(f"\n[bold]📄 Retrieved Chunks for: '{query}'[/bold]\n")
        
        for i, chunk_data in enumerate(relevant_chunks, 1):
            chunk = chunk_data['chunk']
            metadata = chunk_data['metadata']
            score = chunk_data['similarity_score']
            
            console.print(Panel(
                f"[bold]File:[/bold] {metadata['filename']}\n"
                f"[bold]Chunk ID:[/bold] {metadata['chunk_id']}/{metadata['total_chunks']}\n"
                f"[bold]Similarity Score:[/bold] {score:.4f}\n"
                f"[bold]Length:[/bold] {len(chunk)} characters\n\n"
                f"[dim]{chunk}[/dim]",
                title=f"Chunk {i}",
                border_style="cyan"
            ))
    
    def interactive_mode(self):
        """Run interactive chat session."""
        # Display welcome message
        console.print(Panel.fit(
            "[bold blue]🤖 RAG Chatbot - Interactive Mode[/bold blue]\n\n"
            "Ask questions about your documents!\n\n"
            "[dim]Commands:\n"
            "  'quit' or 'exit' - Exit the chat\n"
            "  'info' - Show database information\n"
            "  'debug <query>' - Show retrieved chunks for a query[/dim]",
            border_style="blue"
        ))
        
        # Show database info
        db_info = self.pipeline.get_database_info()
        console.print(f"[green]📚 Loaded {db_info['total_documents']} documents "
                     f"with {db_info['total_chunks']} chunks[/green]\n")
        
        while True:
            try:
                # Get user input
                query = Prompt.ask("\n[bold cyan]❓ Your question[/bold cyan]").strip()
                
                if not query:
                    continue
                
                # Handle commands
                if query.lower() in ['quit', 'exit', 'q']:
                    console.print("[yellow]👋 Goodbye![/yellow]")
                    break
                
                elif query.lower() == 'info':
                    self.show_info()
                    continue
                
                elif query.lower().startswith('debug '):
                    debug_query = query[6:].strip()
                    if debug_query:
                        self.display_chunk_details(debug_query)
                    continue
                
                # Process the question
                console.print()  # Add space
                response = self.search_and_respond(query)
                self.display_response(query, response)
                
            except KeyboardInterrupt:
                console.print("\n[yellow]👋 Goodbye![/yellow]")
                break
            except Exception as e:
                console.print(f"\n[red]❌ Error: {str(e)}[/red]")
    
    def show_info(self):
        """Show database information."""
        info = self.pipeline.get_database_info()
        
        info_table = Table(title="📊 Database Information", border_style="blue")
        info_table.add_column("Property", style="cyan")
        info_table.add_column("Value", style="green")
        
        for key, value in info.items():
            info_table.add_row(str(key).replace('_', ' ').title(), str(value))
        
        console.print(info_table)


@click.command()
@click.option('--vector-db', default=None, help=f'Path to vector database (default: {config.VECTOR_DB_PATH})')
@click.option('--num-results', default=5, help='Number of relevant chunks to retrieve')
@click.option('--query', '-q', default=None, help='Single query mode (non-interactive)')
@click.option('--debug', is_flag=True, help='Show retrieved chunks in single query mode')
def main(vector_db, num_results, query, debug):
    """
    RAG Chatbot - Ask questions about your documents.
    
    This chatbot uses the vector database created by train.py to find relevant
    document chunks and generate answers using OpenAI's GPT models.
    """
    
    # Initialize chatbot
    chatbot = RAGChatbot(vector_db, num_results)
    
    # Single query mode
    if query:
        console.print(f"[bold]❓ Question:[/bold] {query}\n")
        
        if debug:
            chatbot.display_chunk_details(query)
            console.print("\n" + "="*50 + "\n")
        
        response = chatbot.search_and_respond(query)
        chatbot.display_response(query, response)
        return
    
    # Interactive mode
    chatbot.interactive_mode()


if __name__ == "__main__":
    main()