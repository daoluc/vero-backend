from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import chromadb
from chromadb.config import Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from dotenv import load_dotenv
import unicodedata

app = FastAPI(
    title="Vero API",
    description="API for searching and retrieving information from internal documents",
    version="1.0.0"
)

# Load environment variables
load_dotenv()

# Initialize ChromaDB client
chroma_client = chromadb.Client(Settings(
    persist_directory="chroma_db",
    is_persistent=True
))
embeddings_collection = chroma_client.get_or_create_collection("embeddings")
vector_store = ChromaVectorStore(chroma_collection=embeddings_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

class SearchQuery(BaseModel):
    query: str
    top_k: int = 3
    
class Chunk(BaseModel):
    text: str
    source: str
    file_name: str
class SearchResponse(BaseModel):
    results: List[Chunk]

@app.post("/internal_search", response_model=SearchResponse)
async def internal_search(search_query: SearchQuery):
    """
    Search for similar chunks in the processed documents based on the query.
    
    Args:
        query: The search query containing the text to search for and number of results to return
        
    Returns:
        A list of text chunks that are most similar to the query
    """
    try:
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context,
        )
        
        # Use retriever instead of query engine
        retriever = index.as_retriever(similarity_top_k=search_query.top_k)
        source_nodes = retriever.retrieve(search_query.query)
        
        # Extract text content from the source nodes and normalize Unicode characters
        results = [
            {
                'text': unicodedata.normalize('NFKC', node.text), 
                'source': node.metadata.get('source', 'Unknown'), 
                'file_name': node.metadata.get('file_name', 'Unknown')
            } for node in source_nodes]        
        
        return SearchResponse(results=results)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while searching documents: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

#uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
#http://localhost:8000/docs