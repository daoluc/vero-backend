import os
from typing import List
from pathlib import Path
from dotenv import load_dotenv
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
    StorageContext,
)
from llama_index.vector_stores import ChromaVectorStore
import chromadb
from chromadb.config import Settings

load_dotenv()

class PDFProcessor:
    def __init__(self, persist_directory: str = "chroma_db"):
        """Initialize the PDF processor with vector store configuration."""
        self.persist_directory = persist_directory
        self._setup_vector_store()
        
    def _setup_vector_store(self):
        """Set up the Chroma vector store."""
        # Initialize Chroma client
        chroma_client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create or get the collection
        self.embeddings_collection = chroma_client.get_or_create_collection("pdf_embeddings")
        
        # Create vector store
        self.vector_store = ChromaVectorStore(chroma_collection=self.embeddings_collection)
        
        # Create storage context
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        
        # Create service context
        self.service_context = ServiceContext.from_defaults()
    
    def process_pdf(self, local_path: str, folder_id: str) -> None:
        """Process a single PDF file and store its embeddings."""
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"PDF file not found: {local_path}")
            
        if not local_path.lower().endswith('.pdf'):
            raise ValueError(f"File is not a PDF: {local_path}")
        
        # Get file information
        file_name = os.path.basename(local_path)
        
        # Load the PDF
        documents = SimpleDirectoryReader(
            input_files=[local_path]
        ).load_data()
        
        # Add file information to each document's metadata
        for doc in documents:
            doc.metadata["folder_id"] = folder_id
            doc.metadata["file_name"] = file_name
        
        # Create index and store embeddings
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=self.storage_context,
            service_context=self.service_context
        )
        
        print(f"Processed {local_path} successfully")
        
    def get_similar_chunks(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve similar chunks based on a query."""
        # Create a query engine
        index = VectorStoreIndex.from_vector_store(
            self.vector_store,
            service_context=self.service_context
        )
        query_engine = index.as_query_engine()
        
        # Get response
        response = query_engine.query(query)
        return response.source_nodes

if __name__ == "__main__":
    # Example usage
    processor = PDFProcessor()
    # Process a single PDF
    # processor.process_pdf("path/to/your/pdf")
    # Process a directory of PDFs
    # processor.process_directory("path/to/your/pdf/directory") 