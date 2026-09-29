import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Force Langchain's OpenAI clients to route through OpenRouter
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

class Config:
    """
    Configuration class to hold application-level constants and environment variables.
    """
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agentic-ai-index")
    
    # Embedding configurations
    EMBEDDING_MODEL = "models/gemini-embedding-001"
    
    # LLM configurations
    LLM_MODEL = "gemini-3.5-flash"
    LLM_TEMPERATURE = 0.0
    
    # Chunking parameters for Document parsing
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 100
    
    # Retriever parameters
    RETRIEVER_K = 3

    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    DEFAULT_PDF_PATH = os.path.join(DATA_DIR, "Ebook-Agentic-AI.pdf")
