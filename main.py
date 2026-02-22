import os
import pickle
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from langchain_groq import ChatGroq
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# Lazy-load vectorstore and retrieval chain
vectorstore = None
retrieval_chain = None

# Store chat history
chat_history = []

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "data", "vectorstore.pkl")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Groq API key


load_dotenv()  # Loads variables from .env

groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize the Groq LLM
llm = ChatGroq(model_name="meta-llama/llama-4-maverick-17b-128e-instruct")



def load_vectorstore_and_chain():
    global vectorstore, retrieval_chain
    if vectorstore is None:
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)

        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        prompt = ChatPromptTemplate.from_template("""
            You are a precise medical assistant.

            Use ONLY the information inside the context below to answer the question.
            Your answer must be direct and start immediately with the information but be really friendly, make the patient feel good.

            Rules:
            - Do NOT mention the context.
            - Do NOT restate the question.
            - Do NOT explain your reasoning.
            - Do NOT say phrases like "Based on the provided context".
            - If the answer is not in the context, say: "I don't have enough information."

            <context>
            {context}
            </context>

            {input}
            """)
        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain

# Serve the HTML form
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "chat_history": chat_history})

# Handle the form submission
@app.post("/", response_class=HTMLResponse)
def ask(request: Request, question: str = Form(...)):
    chain = load_vectorstore_and_chain()
    response = chain.invoke({"input": question})
    answer = response["answer"]
    
    # Add Q&A to chat history
    chat_history.append({"question": question, "answer": answer})
    
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "chat_history": chat_history}
    )

@app.post("/clear", response_class=HTMLResponse)
def clear(request: Request):
    global chat_history
    chat_history = []  # reset the chat history
    return templates.TemplateResponse("index.html", {"request": request, "chat_history": chat_history})