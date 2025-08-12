# RAG Chatbot System

A locally managed RAG (Retrieval-Augmented Generation) chatbot that processes PDF documents, creates embeddings using local models, and provides intelligent question-answering capabilities.

## Features

- **Local PDF Processing**: Extract text from hundreds of PDF documents
- **Local Embeddings**: Uses sentence-transformers for high-quality embeddings
- **FAISS Vector Database**: Fast similarity search with persistent storage
- **OpenAI Integration**: GPT-powered response generation with RAG context
- **CLI Interface**: Easy-to-use command-line tools for training and chatting
- **Rich Output**: Beautiful terminal interface with color and formatting

## Architecture

1. **PDF Processing** (`pdf_processor.py`): Extracts and cleans text from PDF files
2. **Embeddings** (`embeddings.py`): Generates embeddings using sentence-transformers
3. **Vector Database** (`vector_db.py`): FAISS-based similarity search and storage
4. **Document Pipeline** (`document_pipeline.py`): Orchestrates the entire processing workflow
5. **LLM Client** (`llm_client.py`): OpenAI API integration with RAG prompting
6. **CLI Tools**: Training (`train.py`) and chatbot (`chat.py`) interfaces

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Copy the example environment file and add your OpenAI API key:

```bash
cp .env.example .env
```

Edit `.env` file:
```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-3.5-turbo
CHUNK_SIZE=500
CHUNK_OVERLAP=50
VECTOR_DB_PATH=./vector_db
PDF_DIR=./pdfs
```

### 3. Prepare Your Documents

Create a directory for your PDF files and add your documents:

```bash
mkdir pdfs
# Copy your PDF files to the pdfs/ directory
```

## Usage

### Training (Document Processing)

Process your PDF documents and create the vector database:

```bash
# Basic training
python train.py

# Custom configuration
python train.py --pdf-dir ./my-documents --chunk-size 1000 --chunk-overlap 100

# Show database info
python train.py --info

# Preview processed chunks
python train.py --preview
```

Training will:
1. Extract text from all PDF files
2. Split text into overlapping chunks
3. Generate embeddings using `all-MiniLM-L6-v2`
4. Store embeddings in FAISS index
5. Save everything to disk

### Chatbot Usage

Start the interactive chatbot:

```bash
# Interactive mode
python chat.py

# Single query mode
python chat.py -q "What is the main topic discussed in the documents?"

# Debug mode (show retrieved chunks)
python chat.py -q "Your question" --debug
```

### Interactive Commands

In interactive mode, you can use these commands:

- Ask any question about your documents
- `info` - Show database statistics
- `debug <query>` - Show retrieved chunks for a query
- `quit` or `exit` - Exit the chatbot

## Configuration Options

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `OPENAI_MODEL`: OpenAI model to use (default: gpt-3.5-turbo)
- `CHUNK_SIZE`: Size of text chunks in characters (default: 500)
- `CHUNK_OVERLAP`: Overlap between chunks in characters (default: 50)
- `VECTOR_DB_PATH`: Path to store vector database (default: ./vector_db)
- `PDF_DIR`: Directory containing PDF files (default: ./pdfs)

### Embedding Model

The system uses `all-MiniLM-L6-v2` by default, which provides:
- 384-dimensional embeddings
- Good balance of speed and quality
- Suitable for most document types

You can change this in `config.py` by modifying `EMBEDDING_MODEL`.

## File Structure

```
ask-spurgeon/
├── requirements.txt          # Python dependencies
├── .env.example             # Environment configuration template
├── config.py                # Configuration settings
├── pdf_processor.py         # PDF text extraction
├── embeddings.py            # Local embedding generation
├── vector_db.py             # FAISS vector database
├── document_pipeline.py     # Processing workflow
├── llm_client.py            # OpenAI integration
├── train.py                 # Training CLI
├── chat.py                  # Chatbot CLI
├── pdfs/                    # PDF documents (create this)
└── vector_db/               # Generated vector database
    ├── faiss.index          # FAISS index file
    ├── metadata.json        # Database metadata
    └── chunks.pkl           # Text chunks and metadata
```

## Example Workflow

1. **Setup**: Install dependencies and configure API key
2. **Add Documents**: Place PDF files in `pdfs/` directory
3. **Train**: Run `python train.py` to process documents
4. **Chat**: Run `python chat.py` to start asking questions

## Performance Tips

- **Chunk Size**: Smaller chunks (300-500) for specific questions, larger chunks (800-1200) for broader topics
- **Number of Results**: Use 3-5 chunks for focused answers, 5-10 for comprehensive responses
- **Token Management**: The system automatically manages token limits for OpenAI API calls

## Troubleshooting

### Common Issues

1. **"No documents found"**: Ensure PDF files are in the correct directory
2. **"OpenAI API key not provided"**: Check your `.env` file configuration
3. **"No database found"**: Run training first with `python train.py`
4. **High token usage**: Reduce chunk size or number of results

### Debug Mode

Use debug mode to see what chunks are being retrieved:

```bash
python chat.py -q "Your question" --debug
```

This shows the exact text chunks used to generate the answer.

## Advanced Usage

### Custom PDF Directory

```bash
python train.py --pdf-dir /path/to/your/documents
```

### Force Reprocessing

```bash
python train.py --force
```

### Different Vector Database Location

```bash
python train.py --vector-db /path/to/vector/db
python chat.py --vector-db /path/to/vector/db
```

## Dependencies

- `sentence-transformers`: Local embedding generation
- `faiss-cpu`: Vector similarity search
- `pymupdf`: PDF text extraction
- `openai`: GPT API integration
- `click`: CLI interface
- `rich`: Terminal formatting
- `python-dotenv`: Environment configuration

## License

This project is open source and available under the MIT License.