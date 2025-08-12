import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./vector_db")
PDF_DIR = os.getenv("PDF_DIR", "./pdfs")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"