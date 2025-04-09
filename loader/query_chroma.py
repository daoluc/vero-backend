import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from chromadb.config import Settings as ChromaSettings

load_dotenv()

def query_chroma_db(persist_directory: str = "chroma_db", query_text: str = None, top_k: int = 5):
    """
    Query the Chroma database for similar content.
    
    Args:
        persist_directory: Directory where Chroma database is stored
        query_text: The text to query for
        top_k: Number of results to return
    """
    # Initialize Chroma client
    chroma_client = chromadb.PersistentClient(
        path=persist_directory,
        settings=ChromaSettings(anonymized_telemetry=False)
    )
    
    # Get the embeddings collection
    embeddings_collection = chroma_client.get_or_create_collection("pdf_embeddings")
    
    # Create vector store
    vector_store = ChromaVectorStore(chroma_collection=embeddings_collection)
    
    # Create storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Create index from vector store
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context
    )
    
    # Create query engine
    query_engine = index.as_query_engine(similarity_top_k=top_k)
    
    # If no query text is provided, use a default query
    if not query_text:
        query_text = "What documents have been processed?"
    
    # Get response
    response = query_engine.query(query_text)
    
    # Print the response
    print(f"\nQuery: {query_text}")
    print(f"\nResponse: {response.response}")
    
    # Print source nodes (chunks) with their metadata
    print("\nSource chunks:")
    for i, node in enumerate(response.source_nodes):
        print(f"\nChunk {i+1}:")
        print(f"Score: {node.score}")
        print(f"File: {node.metadata.get('file_name', 'Unknown')}")
        print(f"Path: {node.metadata.get('file_path', 'Unknown')}")
        print(f"Content: {node.text}...")  # Print first 200 chars of content

if __name__ == "__main__":
    # Example usage
    # You can provide your own query or use the default
    query_chroma_db(query_text="What is Bakra Beverage")
    
    # To see all documents that have been processed
    # query_chroma_db(query_text="List all documents that have been processed")
    
    # To search for specific content
    # query_chroma_db(query_text="What does the document say about machine learning?") 