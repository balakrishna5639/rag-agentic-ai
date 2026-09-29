# 🤖 RAG-Based AI Chatbot: Agentic AI Assistant

**Live API Demo:** [https://rag-agentic-ai-jlk5.onrender.com/docs](https://rag-agentic-ai-jlk5.onrender.com/docs)

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-0.1.0-green)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-teal)
![Pinecone](https://img.shields.io/badge/Pinecone-VectorDB-blueviolet)

**Objective:** Develop a robust Retrieval-Augmented Generation (RAG) chatbot using Python, LangGraph, Pinecone, and FastAPI. This system strictly grounds its answers based on the provided "Agentic AI" eBook.

---

## 📑 Table of Contents
1. [Project Overview](#-project-overview)
2. [System Architecture](#-system-architecture)
3. [Prerequisites](#-prerequisites)
4. [Installation & Setup](#-installation--setup)
5. [How to Run](#-how-to-run)
6. [Testing & Quality Assurance](#-testing--quality-assurance)
7. [AI Tools Used](#-ai-tools-used)
8. [Known Limitations](#-known-limitations)

---

## 🚀 Project Overview
This project is an end-to-end RAG architecture implementation designed to parse, index, and retrieve domain-specific information from a PDF document. By combining **LangGraph** for stateful orchestration, **Pinecone** for scalable vector storage, and **Google Gemini** for generation, the chatbot effectively answers user queries while preventing hallucinations via strict context-grounding.

### 🌟 Key Features
- **Semantic Search:** Employs Gemini embeddings (`models/gemini-embedding-001`) for dense vector generation.
- **Stateful Graph Orchestration:** Manages data flow between retrieval, generation, and hallucination-grading nodes using `langgraph`.
- **Strict Grounding:** Refuses to answer queries (e.g., "What is the capital of France?") if the knowledge is absent in the document context.
- **RESTful API:** Exposes endpoints via FastAPI returning exact structured JSON (query, final_answer, retrieved_context_chunks, confidence_score).

### 📄 Verified Response Structure
Every API request to `/chat` returns a strictly formatted JSON payload:
```json
{
  "query": "What is Agentic AI?",
  "final_answer": "Agentic AI refers to autonomous systems that...",
  "retrieved_context_chunks": [
    "Chunk 1 text from PDF...",
    "Chunk 2 text from PDF..."
  ],
  "confidence_score": 0.92
}
```

---

## 🧠 System Architecture

```text
+-------------------+       +-------------------+       +--------------------+
|                   |       |                   |       |                    |
|  1. PDF Document  +------>+  2. PyPDFLoader & +------>+ 3. Gemini Embed.   |
|   (Agentic AI)    |       |   TextSplitter    |       |    (Vector Gen)    |
|                   |       |                   |       |                    |
+-------------------+       +-------------------+       +---------+----------+
                                                                  |
                                                                  v
+-------------------+       +-------------------+       +---------+----------+
|                   |       |                   |       |                    |
|  6. FastAPI JSON  +<------+  5. LangGraph RAG |<------+ 4. Pinecone Vector |
|     Response      |       |   (State Graph)   |       |    Database Index  |
|                   |       |                   |       |                    |
+-------------------+       +--------^----------+       +--------------------+
                                     |
                                     |
                          +----------+----------+
                          |                     |
                          |  - Retrieve Node    |
                          |  - Generate Node    |
                          |  - Grade Node       |
                          |                     |
                          +---------------------+
```

---

## ⚙️ Prerequisites
Ensure your system meets the following specifications:
- **Python:** `v3.10` or higher
- **API Keys:** You will need valid keys for:
  - Google Gemini (LLM and Embeddings)
  - Pinecone (Vector Index)

---

## 🛠️ Installation & Setup

**Step 1: Clone the Repository**
```bash
git clone https://github.com/balakrishna5639/rag-agentic-ai
cd rag-agentic-ai
```

**Step 2: Create a Virtual Environment**
```bash
python -m venv venv
# For Windows:
venv\Scripts\activate
# For macOS/Linux:
source venv/bin/activate
```

**Step 3: Install Dependencies**
```bash
pip install -r requirements.txt
```

**Step 4: Environment Variables**
Copy the template `.env.example` to a new file named `.env` and fill in your credentials. Ensure you have a Pinecone index named `agentic-ai-index` with dimensions set to `3072` and metric `cosine`.

---

## 🚀 How to Run

Before testing the chatbot, you must populate the vector database with the eBook's content. The ingestion script will automatically download the eBook PDF and index it.

**1. Run Data Ingestion:**
```bash
python -m src.ingestion
```

**2. Start the FastAPI Server:**
```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```
Once running, navigate to the auto-generated interactive Swagger UI: 👉 **http://127.0.0.1:8000/docs**

### 🌐 How to Use the Swagger UI
1. Open the [Live API Demo](https://rag-agentic-ai-jlk5.onrender.com/docs) (or your local `http://127.0.0.1:8000/docs`).
2. Click on the green **`POST /chat`** endpoint bar to expand it.
3. Click the **"Try it out"** button on the right side.
4. In the **Request body** text box, edit the `"query"` string to ask any question about Agentic AI.
5. Click the large blue **"Execute"** button.
6. Scroll down to the **"Server response"** section to see the fully structured JSON answer, complete with confidence scores and context chunks!

---

## 🧪 Testing & Quality Assurance

A dedicated testing script (`tests_sample_queries.py`) is included to run benchmarking queries and validate context-grounding mechanisms.

Ensure your FastAPI server is running in a separate terminal, then execute:
```bash
python tests_sample_queries.py
```

### 📊 Benchmark Queries Evaluated:
1. *What is the core definition of Agentic AI as outlined in the eBook?*
2. *What are the main architectural components required to build agentic systems?*
3. *What real-world industry use cases for Agentic AI are discussed in the eBook?*
4. *How does Agentic AI differ from traditional generative AI chatbots according to the text?*
5. *What key challenges or limitations of Agentic AI are mentioned in the document?*
6. *What is the capital of France?* (Expected: Graceful Refusal / Confidence = 0.0)

---

## 🤖 AI Tools Used
This project was developed with the assistance of **Google Gemini (Antigravity IDE)** for rapid prototyping, architecture scaffolding, code refactoring, and debugging.

## ⚠️ Known Limitations
- **Ingestion Speed:** Due to API rate limits with embedding providers, large PDF chunk ingestion might take several minutes.
- **Grader Latency:** The hallucination grading step adds a second LLM inference call, marginally increasing the overall response latency compared to a single-shot generation RAG.
- **Cosine Accuracy:** Extremely subtle conceptual differences might bypass the semantic retrieval if the user's phrasing is heavily disconnected from the document's vocabulary.
