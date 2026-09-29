import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import os
import logging
from src.graph import build_rag_graph
from src.config import Config

# API metadata
app = FastAPI(
    title="RAG-based AI Chatbot API",
    description="An intelligent Agentic AI retrieval augmented generation (RAG) backend utilizing LangGraph and Pinecone.",
    version="1.0.0"
)

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialization flag and global graph variable to avoid recompiling on every request
graph = None

@app.on_event("startup")
async def startup_event():
    global graph
    logger.info("Initializing LangGraph RAG pipeline...")
    try:
        graph = build_rag_graph(index_name=Config.PINECONE_INDEX_NAME)
        logger.info("LangGraph pipeline successfully loaded.")
    except Exception as e:
        logger.error(f"Failed to load Graph: {str(e)}")
        # Note: We won't crash the server here so it can show the swagger UI, 
        # but queries will fail if API keys are missing.

class QueryRequest(BaseModel):
    query: str = Field(..., example="What are the core components of an Agentic Architecture?")

class QueryResponse(BaseModel):
    query: str
    final_answer: str
    retrieved_context_chunks: List[str]
    confidence_score: float

@app.get("/")
def read_root():
    return {"message": "Welcome to the RAG AI Chatbot API. Navigate to /docs for the interactive API documentation."}

@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    """
    REST endpoint to query the RAG chatbot.
    Takes a JSON body containing the question and returns the answer, context, and score.
    """
    if graph is None:
        raise HTTPException(status_code=500, detail="Graph not initialized. Check your API keys and configuration.")
        
    logger.info(f"Received query: {request.query}")
    
    # Base state payload
    initial_state = {
        "question": request.query,
        "context": [],
        "answer": "",
        "score": 0.0
    }
    
    try:
        # Execute stateful graph
        result = graph.invoke(initial_state)
        
        return QueryResponse(
            query=request.query,
            final_answer=result["answer"],
            retrieved_context_chunks=result["context"],
            confidence_score=result["score"]
        )
    except Exception as e:
        logger.error(f"Error during query execution: {str(e)}")
        raise HTTPException(status_code=500, detail="An internal error occurred while processing the request.")

if __name__ == "__main__":
    # Standard boilerplate for running the server locally via the python execution command
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
