import os  
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader  
from langchain.text_splitter import RecursiveCharacterTextSplitter  
from langchain_community.embeddings import HuggingFaceEmbeddings  
from langchain_community.vectorstores import Qdrant  
from qdrant_client import QdrantClient  

DATA_DIR = "F:/Wearables/Medical-RAG-LLM/Data"  

def ingest_docs():
    try:
        
        loader = DirectoryLoader(
            DATA_DIR,
            glob="**/*.pdf",  # in directory + subdirectory
            loader_cls=PyPDFLoader
        )
        documents = loader.load() 
        print(f"Loading documents from: {DATA_DIR}")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # characters
            chunk_overlap=100  
        )
        texts = text_splitter.split_documents(documents) 
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",  # text to vectors
            model_kwargs={'device': 'cpu'} 
        )
        
        #creating Qdrant collections
        from qdrant_client.http.models import VectorParams

        client = QdrantClient("http://localhost:6333", timeout=60)  
        collection_name = "medical_docs"
        
        try:
            client.delete_collection(collection_name=collection_name)  
            print(f"Deleted existing collection: {collection_name}")
        except:
            print(f"No existing collection to delete: {collection_name}")  
            
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=384,  
                distance="Cosine" 
            )
        )
        print(f"Created fresh collection: {collection_name}")
        db = Qdrant(
            client=client,
            embeddings=embeddings,
            collection_name=collection_name
        )
        
        batch_size = 10  #10 chunks at a time
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]  
            db.add_documents(batch)  
            print(f"Processed batch {i // batch_size + 1} of {len(texts) // batch_size + 1}")
        
        print(f"Successfully processed {len(documents)} medical documents") 
        return True
        
    except Exception as e:
        print(f"Error in document ingestion: {e}") 
        return False

if __name__ == "__main__":
    ingest_docs()  
