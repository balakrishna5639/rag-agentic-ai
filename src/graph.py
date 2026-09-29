import logging
from typing import List, TypedDict, Dict, Any
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from src.config import Config
import json

logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    """
    Type definition representing the state of our execution graph.
    """
    question: str
    context: List[str]
    answer: str
    score: float
    retrieval_scores: List[float]
    is_grounded: bool

def build_rag_graph(index_name: str = Config.PINECONE_INDEX_NAME):
    embeddings = GoogleGenerativeAIEmbeddings(
        model=Config.EMBEDDING_MODEL,
        google_api_key=Config.GOOGLE_API_KEY
    )
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    llm = ChatGoogleGenerativeAI(
        model=Config.LLM_MODEL, 
        temperature=Config.LLM_TEMPERATURE,
        google_api_key=Config.GOOGLE_API_KEY
    )

    def retrieve_node(state: AgentState):
        """
        Retrieves top-k chunks from Pinecone and extracts cosine similarity scores.
        """
        logger.info(f"Retrieving context for query: {state['question']}")
        
        # Perform similarity search with score
        docs_and_scores = vectorstore.similarity_search_with_score(
            state["question"], 
            k=Config.RETRIEVER_K
        )
        
        context_texts = []
        scores = []
        for doc, score in docs_and_scores:
            context_texts.append(doc.page_content)
            scores.append(float(score))
            
        return {"context": context_texts, "retrieval_scores": scores}

    def generate_node(state: AgentState):
        """
        Generates an answer based ONLY on retrieved context.
        """
        logger.info("Generating response based on retrieved context...")
        context_str = "\n\n".join(state["context"])
        
        prompt = f"""You are an expert AI assistant. Answer the user's question based strictly on the provided context. 
If the context doesn't contain the answer, say "I don't know." Do not hallucinate.

Context:
{context_str}

Question: {state['question']}
Answer:"""

        response = llm.invoke(prompt)
        content = response.content
        if isinstance(content, list):
            content = " ".join([c if isinstance(c, str) else c.get("text", "") for c in content])
        return {"answer": str(content).strip()}

    def grade_hallucination_node(state: AgentState):
        """
        Evaluates whether the generated answer is grounded in the retrieved context.
        Handles out-of-scope queries by outputting the specific fallback message.
        """
        logger.info("Grading answer for groundedness...")
        
        # If the LLM already admitted it doesn't know, it's out of scope
        if "I don't know" in state["answer"] or "I do not know" in state["answer"]:
            return {
                "answer": "This information is not available in the provided document.",
                "score": 0.0,
                "is_grounded": False
            }

        context_str = "\n\n".join(state["context"])
        
        # Use LLM as a grader to verify groundedness
        grader_prompt = f"""You are a grader evaluating whether an answer is grounded in the context.
Context:
{context_str}

Answer:
{state['answer']}

Is the answer strictly based on the facts in the Context? 
Answer 'yes' or 'no' only."""
        
        grade_response = llm.invoke(grader_prompt)
        content = grade_response.content
        if isinstance(content, list):
            content = " ".join([c if isinstance(c, str) else c.get("text", "") for c in content])
        grade = str(content).strip().lower()
        
        if "yes" in grade:
            # Calculate average cosine similarity from retrieval as the confidence score
            avg_score = sum(state["retrieval_scores"]) / len(state["retrieval_scores"]) if state["retrieval_scores"] else 0.0
            # Ensure score is normalized or maxed at 1.0 (Pinecone cosine can occasionally be slightly > 1.0 due to float math)
            confidence = min(avg_score, 1.0)
            return {"is_grounded": True, "score": round(confidence, 4)}
        else:
            return {
                "answer": "This information is not available in the provided document.",
                "score": 0.0,
                "is_grounded": False
            }

    # ------------------ GRAPH ASSEMBLY ------------------
    logger.info("Assembling LangGraph workflow...")
    workflow = StateGraph(AgentState)
    
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("grade_hallucination", grade_hallucination_node)
    
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "grade_hallucination")
    workflow.add_edge("grade_hallucination", END)
    
    return workflow.compile()
