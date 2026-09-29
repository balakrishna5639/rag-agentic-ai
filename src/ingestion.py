import os
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from src.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_ingestion(pdf_path: str = Config.DEFAULT_PDF_PATH, index_name: str = Config.PINECONE_INDEX_NAME):
    """
    Handles the ETL (Extract, Transform, Load) pipeline for the document.
    Reads the PDF, splits it into semantic chunks, generates embeddings, 
    and upserts them to the Pinecone Vector Database.
    """
    if not os.path.exists(pdf_path):
        logger.info(f"PDF not found at {pdf_path}. Downloading from Konverge...")
        import urllib.request
        pdf_url = "https://konverge.ai/pdf/Ebook-Agentic-AI.pdf"
        try:
            os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
            urllib.request.urlretrieve(pdf_url, pdf_path)
            logger.info("Download completed successfully.")
        except Exception as e:
            logger.error(f"Failed to download PDF: {e}")
            raise FileNotFoundError(f"PDF document not found and could not be downloaded: {pdf_path}")

    logger.info(f"Starting ingestion process for: {pdf_path}")
    
    try:
        # Step 1: Data Extraction
        logger.info("Extracting text from PDF...")
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        logger.info(f"Successfully loaded {len(docs)} pages.")

        # Step 2: Data Transformation (Chunking)
        logger.info(f"Chunking text (Size: {Config.CHUNK_SIZE}, Overlap: {Config.CHUNK_OVERLAP})...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        chunks = text_splitter.split_documents(docs)
        logger.info(f"Document split into {len(chunks)} text chunks.")

        # Step 3: Data Loading (Embeddings & Vector Store)
        logger.info("Initializing OpenAI Embeddings...")
        embeddings = OpenAIEmbeddings(model=Config.EMBEDDING_MODEL)
        
        logger.info(f"Upserting vectors into Pinecone Index: {index_name}...")
        vector_store = PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            index_name=index_name
        )
        logger.info("Ingestion completed successfully.")
        
        return vector_store
    except Exception as e:
        logger.error(f"An error occurred during ingestion: {str(e)}")
        raise e

if __name__ == "__main__":
    # Test script execution for ingestion module
    run_ingestion()
