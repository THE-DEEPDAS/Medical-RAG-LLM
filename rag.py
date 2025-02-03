# Remove medical imports and add financial ones
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles 
from langchain_community.llms import CTransformers
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.chains import RetrievalQA
from qdrant_client import QdrantClient
import os
import json
from typing import Optional, Dict, Any
from Data.insurance_data import insurance_data
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

# Initialize the financial query prompt template
FINANCIAL_QUERY_PROMPT = """
You are a financial advisor assistant specializing in insurance and banking. Use the following information to answer the query.

Context: {context}
Query: {query}

Please provide a clear and detailed response focusing on the financial information requested.
If comparing products or policies, highlight key differences in features, costs, and benefits.
If the query is not related to available financial information, politely indicate that.

Response:
"""

# Update model path
MODEL_PATH = "F:/Wearables/Medical-RAG-LLM/models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"

# Initialize components
try:
    # Use local model directly
    llm = CTransformers(
        model=MODEL_PATH,
        model_type="mistral",
        config=config
    )
    print("Successfully loaded local model from:", MODEL_PATH)
    
    # Initialize embeddings with specific kwargs
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    client = QdrantClient("http://localhost:6333")
    
    # Create vector store for financial documents
    db = Qdrant(
        client=client, 
        embeddings=embeddings,
        collection_name="financial_docs"
    )
    
    retriever = db.as_retriever(search_kwargs={"k": 3})
    
except Exception as e:
    print(f"Initialization error: {e}")
    print(f"Make sure the model exists at: {MODEL_PATH}")
    raise

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/query")
async def process_query(request: QueryRequest):
    """Handle financial queries"""
    try:
        query = request.query
        # Get relevant documents
        docs = retriever.invoke(query)
        context = "\n".join([doc.page_content for doc in docs])
        
        # Add insurance data to context
        insurance_context = json.dumps(insurance_data, indent=2)
        combined_context = f"{context}\n\nInsurance Information:\n{insurance_context}"
        
        # Ensure the combined context does not exceed the maximum context length
        max_context_length = config['context_length']
        if len(combined_context) > max_context_length:
            combined_context = combined_context[:max_context_length]
        
        # Format prompt
        prompt = FINANCIAL_QUERY_PROMPT.format(
            context=combined_context,
            query=query
        )
        
        # Get response from LLM
        response = llm.invoke(prompt)
        
        return JSONResponse(content={
            "query": query,
            "response": response.strip()
        })
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

# Add helper function to search financial info
def search_financial_info(query: str) -> dict:
    """Search through financial documents and insurance data"""
    results = []
    query = query.lower()
    
    # Search in insurance data
    for company, company_data in insurance_data["companies"].items():
        for insurance_type, type_data in company_data["types"].items():
            for policy_name, policy_data in type_data["policies"].items():
                if (query in company.lower() or
                    query in insurance_type.lower() or
                    query in policy_name.lower()):
                    results.append({
                        "type": "insurance",
                        "company": company,
                        "product": policy_name,
                        "details": policy_data
                    })
    
    # Add results from vector store
    docs = retriever.invoke(query)
    for doc in docs:
        results.append({
            "type": "document",
            "content": doc.page_content,
            "source": doc.metadata.get("source", "Unknown")
        })
    
    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
