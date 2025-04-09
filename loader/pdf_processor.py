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
from db_manager import DatabaseManager

load_dotenv()

class PDFProcessor:
    def __init__(self, persist_directory: str = "chroma_db"):
        """Initialize the PDF processor with vector store configuration."""
        self.persist_directory = persist_directory
        self._setup_vector_store()
        self.db_manager = DatabaseManager()
        
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
    
    def process_pdf(self, local_path: str, folder_id: str) -> bool:
        """
        Process a single PDF file and store its embeddings.
        Returns True if the file was processed, False if it was skipped due to duplication.
        """
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"PDF file not found: {local_path}")
            
        if not local_path.lower().endswith('.pdf'):
            raise ValueError(f"File is not a PDF: {local_path}")
        
        # Calculate content hash
        content_hash = self.db_manager.calculate_file_hash(local_path)
        
        # Check if file has already been processed
        if self.db_manager.is_file_processed(content_hash):
            print(f"Skipping {local_path} - already processed")
            return False
        
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
        
        # Mark file as processed in the database
        self.db_manager.mark_file_processed(local_path, folder_id, content_hash)
        
        print(f"Processed {local_path} successfully")
        return True
        
    def get_similar_chunks(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve similar chunks based on a query without generating response."""
        index = VectorStoreIndex.from_vector_store(
            self.vector_store,
            service_context=self.service_context
        )
        
        # Use retriever instead of query engine
        retriever = index.as_retriever(similarity_top_k=top_k)
        source_nodes = retriever.retrieve(query)
        
        return source_nodes

if __name__ == "__main__":
    # Example usage
    processor = PDFProcessor()
    # Process a single PDF
    # processor.process_pdf("path/to/your/pdf")
    # Process a directory of PDFs
    # processor.process_directory("path/to/your/pdf/directory") 