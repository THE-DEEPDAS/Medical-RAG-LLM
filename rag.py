from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import JSONResponse 
from fastapi.templating import Jinja2Templates  
from langchain_community.llms import CTransformers
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant 
from qdrant_client import QdrantClient 
import os  
import numpy as np    
from pydantic import BaseModel  # For request/response models

class QueryRequest(BaseModel):
    query: str 

app = FastAPI()
# Initialize Jinja2 templates for rendering HTML
templates = Jinja2Templates(directory="templates")

config = {
    'max_new_tokens': 1024,  
    'context_length': 1024,  
    'temperature': 0.1,  # Lower temperature = more deterministic responses, isliye 0.1 rakha hai
    'top_p': 0.9,  # probablity of next token
    'stream': False,  # return full response at once
    'threads': min(4, int(os.cpu_count() / 2)), 
}

MEDICAL_QUERY_PROMPT = """
You are a financial assistant specializing in providing detailed information about investments. Use the following information to answer the query.

Context: {context}
Query: {query}

Please provide the following details in a structured format, with each section on a new line and exactly as labeled:
Investment Type: [specify the type of investment]
Risk Level: [describe the risk level]
Expected Returns: [provide expected returns]
Time Horizon: [specify recommended investment duration]
Key Benefits: [list main benefits]
Potential Drawbacks: [list main drawbacks]
Eligibility: [specify eligibility criteria]

Response:
"""

MODEL_PATH = "F:\Wearables\Medical-RAG-LLM\model"

try:
    # Ctransformers is python wrapper for Mistral LLM
    llm = CTransformers(
        model=MODEL_PATH,
        model_type="mistral",  
        config=config
    )
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},  
        encode_kwargs={'normalize_embeddings': True}  # normalize kiya for better similarity search
    )
    
    client = QdrantClient("http://localhost:6333")
    
    db = Qdrant(
        client=client,
        embeddings=embeddings,
        collection_name="medical_docs"  
    )
    
    retriever = db.as_retriever(search_kwargs={"k": 3})  
except Exception as e:
    print(f"Initialization error is there in retriever: {e}")
    raise HTTPException(status_code=500, detail="Error initializing model or vector store")  

@app.post("/query_new")
async def process_query_new(request: QueryRequest):
    try:
        query = request.query.strip() 
        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty") 
        
        docs = retriever.invoke(query)
        context = "\n".join([doc.page_content for doc in docs]) if docs else ""  
        
        prompt = MEDICAL_QUERY_PROMPT.format(context=context, query=query)
        response = llm.invoke(prompt) 
        
        if not context:
            response += "\n(Note: The response is based solely on the LLM's knowledge.)"
        
        return JSONResponse(content={
            "query": query,
            "response": response.strip(),
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
    return Response(status_code=204)  # No content response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)