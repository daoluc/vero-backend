from mcp.server.fastmcp import FastMCP
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize MCP server
mcp = FastMCP("ChromaDB Search Server")

# Initialize ChromaDB client
chroma_client = chromadb.Client(Settings(
    persist_directory="chroma_db",
    is_persistent=True
))

@mcp.tool()
def search_documents(query: str, n_results: int = 5) -> Dict[str, Any]:
    """
    Search for related information from internal files using ChromaDB.
    
    Args:
        query: The search query to find relevant documents
        n_results: Number of results to return (default: 5)
        
    Returns:
        Dictionary containing documents, metadata, and distances
    """
    try:
        # Get the collection (assuming it's named "documents")
        collection = chroma_client.get_collection("documents")
        
        # Perform the search
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        return {
            "documents": results["documents"][0],
            "metadatas": results["metadatas"][0],
            "distances": results["distances"][0]
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    mcp.run() 