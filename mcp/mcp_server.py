from mcp.server.fastmcp import FastMCP
import chromadb
from chromadb.config import Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from typing import List, Dict, Any
from dotenv import load_dotenv
import unicodedata

# Load environment variables
load_dotenv()

# Initialize MCP server
mcp = FastMCP("Vero Search Server")

# Initialize ChromaDB client
chroma_client = chromadb.Client(Settings(
    persist_directory="chroma_db",
    is_persistent=True
))
embeddings_collection = chroma_client.get_or_create_collection(
    "embeddings",
    configuration={
        "hnsw": {
            "num_threads": 2
        }
    }
)
vector_store = ChromaVectorStore(chroma_collection=embeddings_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

@mcp.tool()
def search_documents(query: str, n_results: int = 3) -> Dict[str, Any]:
    """
    Search for related information from internal documents.
    
    Args:
        query: The search query to find relevant documents
        n_results: Number of results to return (default: 3)
        
    Returns:
        Dictionary containing documents, metadata, and distances
    """
    try:
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context,
        )
        
        # Use retriever instead of query engine
        retriever = index.as_retriever(similarity_top_k=n_results)
        source_nodes = retriever.retrieve(query)
        
        # Extract text content from the source nodes and normalize Unicode characters
        chunk_texts = [unicodedata.normalize('NFKC', node.text) for node in source_nodes]
        
        return chunk_texts
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    mcp.run(transport='stdio')
    # print(search_documents("What is Bakra Beverage?"))