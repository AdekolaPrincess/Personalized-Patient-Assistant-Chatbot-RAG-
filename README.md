# 💊 Personalized Patient Assistant

This is an AI-powered patient chatbot built with FastAPI and a vector database.  
It answers medical-related questions strictly based on uploaded document context using Retrieval-Augmented Generation (RAG).

---


## Objectives

The main objectives of this project are:

- To implement a document-based medical assistant using RAG.
- To ensure responses are generated strictly from trusted medical documents.
- To prevent hallucinations by restricting the model to contextual information only.
- To design a simple and user-friendly chatbot interface.

---

##  System Architecture

The system consists of the following components:

###  Document Embedding
- Medical documents are converted into vector embeddings using Sentence Transformers.
- These embeddings are stored in a persistent vector store (`vectorstore.pkl`).

### Retrieval Mechanism
- When a user submits a question, relevant document chunks are retrieved using similarity search.

###  Response Generation
- The retrieved context is passed into a structured prompt.
- The LLM generates a response strictly based on the retrieved context.
- If the answer is not present in the documents, the system responds:
  
  > "I don't have enough information."

### Web Interface
- Backend: FastAPI
- Frontend: HTML & CSS
- Chat history is maintained per session.
- Includes a "Clear All" feature to reset conversation history.

---

## 🏗️ Tech Stack

- Python
- FastAPI
- Uvicorn
- LangChain
- Sentence Transformers
- LLaMA 4 Maverick (via Groq API)
- Pickle (for vector store persistence)
- HTML & CSS (frontend)

---

## 📂 Project Structure
Personalized Patient Assistant/
│
├── main.py  (FastAPI backend)
├── templates/
│ └── index.html  (Chatbot UI)
├── vectorstore.pkl (Stored embeddings)
├── requirements.txt
└── README.md

