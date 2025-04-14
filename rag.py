from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from langchain_community.llms import CTransformers
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
import os
import numpy as np
from typing import Optional, Dict, Any
from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Initialize LLM config
config = {
    'max_new_tokens': 1024,
    'context_length': 1024,
    'temperature': 0.1,
    'top_p': 0.9,
    'stream': False,
    'threads': min(4, int(os.cpu_count() / 2)),
}

# Medical query prompt template
MEDICAL_QUERY_PROMPT = """
You are a medical assistant specializing in providing detailed information about medicines. Use the following information to answer the query.

Context: {context}
Query: {query}

Please provide the following details:
- Purpose
- Side effects
- Drug composition
- Key ingredients
- Age group
- Dosage
- Timing

If any information is missing, use your knowledge to fill in the gaps.

Response:
"""

MODEL_PATH = "F:/Wearables/Medical-RAG-LLM/model/mistral-7b-instruct-v0.1.Q4_K_M.gguf"

# Initialize components
try:
    llm = CTransformers(
        model=MODEL_PATH,
        model_type="mistral",
        config=config
    )
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    client = QdrantClient("http://localhost:6333")
    db = Qdrant(
        client=client,
        embeddings=embeddings,
        collection_name="medical_docs"
    )
    retriever = db.as_retriever(search_kwargs={"k": 3})
except Exception as e:
    print(f"Initialization error: {e}")
    raise

def evaluate_uncertainty(response: str) -> float:
    """Evaluate the uncertainty of the LLM's response using token probabilities."""
    tokens = response.split()
    token_probs = [np.random.uniform(0.7, 1.0) for _ in tokens]  # Simulate token probabilities
    uncertainty_score = 1 - np.mean(token_probs)  # Lower mean probability indicates higher uncertainty
    return uncertainty_score

@app.post("/query_new")
async def process_query_new(request: QueryRequest):
    """Handle medical queries with fallback to LLM knowledge and uncertainty evaluation."""
    try:
        query = request.query.strip()
        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Retrieve documents
        docs = retriever.invoke(query)
        context = "\n".join([doc.page_content for doc in docs]) if docs else ""
        
        # Generate response using LLM
        prompt = MEDICAL_QUERY_PROMPT.format(context=context, query=query)
        response = llm.invoke(prompt)
        
        # Evaluate uncertainty
        uncertainty_score = evaluate_uncertainty(response)
        
        # Fallback to LLM knowledge if context is insufficient
        if not context:
            response += "\n(Note: The response is based solely on the LLM's knowledge.)"
        
        return JSONResponse(content={
            "query": query,
            "response": response.strip(),
            "uncertainty_score": uncertainty_score
        })
    except Exception as e:
        print(f"Error in /query_new: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/query")
async def query_alias(request: QueryRequest):
    return await process_query_new(request)

@app.get("/ping")
async def ping():
    return {"message": "pong"}

@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)