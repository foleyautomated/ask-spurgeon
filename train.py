#!/usr/bin/env python3

import click
import os
import sys
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint
from document_pipeline import DocumentPipeline
import config

console = Console()


@click.command()
@click.option('--pdf-dir', default=None, help=f'Directory containing PDF files (default: {config.PDF_DIR})')
@click.option('--vector-db', default=None, help=f'Path to vector database (default: {config.VECTOR_DB_PATH})')
@click.option('--chunk-size', default=None, type=int, help=f'Chunk size for text splitting (default: {config.CHUNK_SIZE})')
@click.option('--chunk-overlap', default=None, type=int, help=f'Overlap between chunks (default: {config.CHUNK_OVERLAP})')
@click.option('--info', is_flag=True, help='Show database info without processing')
@click.option('--preview', is_flag=True, help='Preview chunks in the database')
@click.option('--force', is_flag=True, help='Force reprocessing even if database exists')
def main(pdf_dir, vector_db, chunk_size, chunk_overlap, info, preview, force):
    """
    Train the RAG system by processing PDFs and creating embeddings.
    
    This script will:
    1. Extract text from all PDFs in the specified directory
    2. Split text into chunks
    3. Generate embeddings using a local sentence-transformer model
    4. Store embeddings in a FAISS vector database
    """
    
    # Display header
    console.print(Panel.fit(
        "[bold blue]RAG Chatbot Training System[/bold blue]\n"
        "Process PDFs and create vector embeddings",
        border_style="blue"
    ))
    
    # Initialize pipeline
    pipeline = DocumentPipeline(pdf_dir, vector_db)
    
    # Handle info request
    if info:
        show_database_info(pipeline)
        return
    
    # Handle preview request
    if preview:
        show_chunk_preview(pipeline)
        return
    
    # Check if PDFs directory exists
    if not os.path.exists(pipeline.pdf_dir):
        console.print(f"[red]Error: PDF directory '{pipeline.pdf_dir}' does not exist[/red]")
        console.print(f"[yellow]Create the directory and add PDF files, or specify a different path with --pdf-dir[/yellow]")
        sys.exit(1)
    
    # Check for PDFs in directory
    pdf_files = [f for f in os.listdir(pipeline.pdf_dir) if f.lower().endswith('.pdf')]
    if not pdf_files:
        console.print(f"[red]Error: No PDF files found in '{pipeline.pdf_dir}'[/red]")
        console.print(f"[yellow]Add PDF files to the directory and try again[/yellow]")
        sys.exit(1)
    
    console.print(f"[green]Found {len(pdf_files)} PDF files in '{pipeline.pdf_dir}'[/green]")
    
    # Check if database already exists
    db_exists = all(os.path.exists(os.path.join(pipeline.vector_db_path, f)) 
                   for f in ["faiss.index", "metadata.json", "chunks.pkl"])
    
    if db_exists and not force:
        console.print(f"[yellow]Database already exists at '{pipeline.vector_db_path}'[/yellow]")
        if not click.confirm("Do you want to recreate it? (This will overwrite existing data)"):
            console.print("[blue]Use --info to see database information or --force to overwrite[/blue]")
            return
    
    # Show configuration
    console.print("\n[bold]Configuration:[/bold]")
    console.print(f"  PDF Directory: {pipeline.pdf_dir}")
    console.print(f"  Vector DB Path: {pipeline.vector_db_path}")
    console.print(f"  Chunk Size: {chunk_size or config.CHUNK_SIZE}")
    console.print(f"  Chunk Overlap: {chunk_overlap or config.CHUNK_OVERLAP}")
    console.print(f"  Embedding Model: {config.EMBEDDING_MODEL}")
    
    # Confirm before processing
    if not click.confirm("\nProceed with processing?"):
        console.print("[yellow]Processing cancelled[/yellow]")
        return
    
    try:
        # Process documents
        console.print("\n[bold green]Starting document processing...[/bold green]")
        pipeline.process_and_index_documents(chunk_size, chunk_overlap)
        
        console.print("\n[bold green]✅ Training completed successfully![/bold green]")
        console.print("[blue]You can now use the chatbot with: python chat.py[/blue]")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Processing interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Error during processing: {str(e)}[/red]")
        sys.exit(1)


def show_database_info(pipeline: DocumentPipeline):
    """Display information about the current database."""
    info = pipeline.get_database_info()
    
    if info.get("status") == "No database available":
        console.print("[red]No database found[/red]")
        console.print(f"[yellow]Run training to create database: python train.py[/yellow]")
        return
    
    # Create info table
    table = Table(title="Database Information", border_style="blue")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    
    for key, value in info.items():
        table.add_row(str(key).replace('_', ' ').title(), str(value))
    
    console.print(table)


def show_chunk_preview(pipeline: DocumentPipeline):
    """Show a preview of chunks in the database."""
    chunks = pipeline.preview_chunks(max_chunks=5)
    
    if not chunks:
        console.print("[red]No chunks found in database[/red]")
        console.print(f"[yellow]Run training to create database: python train.py[/yellow]")
        return
    
    console.print(f"[bold]Preview of {len(chunks)} chunks:[/bold]\n")
    
    for chunk in chunks:
        console.print(Panel(
            f"[bold]File:[/bold] {chunk['filename']}\n"
            f"[bold]Chunk ID:[/bold] {chunk['chunk_id']}\n"
            f"[bold]Length:[/bold] {chunk['chunk_length']} characters\n\n"
            f"[dim]{chunk['chunk_preview']}[/dim]",
            border_style="cyan",
            title=f"Chunk {chunk['index']}"
        ))


if __name__ == "__main__":
    main()