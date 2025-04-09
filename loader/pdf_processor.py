import os
from typing import List
from dotenv import load_dotenv
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
)
from llama_index.vector_stores.chroma.base import ChromaVectorStore
import chromadb
from chromadb.config import Settings as ChromaSettings
from db_manager import DatabaseManager
import unicodedata

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
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        # Create or get the collection
        self.embeddings_collection = chroma_client.get_or_create_collection("embeddings")
        
        # Create vector store
        self.vector_store = ChromaVectorStore(chroma_collection=self.embeddings_collection)
        
        # Create storage context
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
    
    def process_pdf(self, local_path: str, source: str) -> bool:
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
            doc.metadata["source"] = source
            doc.metadata["file_name"] = file_name
            doc.metadata["content_hash"] = content_hash
        
        # Create index and store embeddings
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=self.storage_context,
        )
        
        # Mark file as processed in the database
        self.db_manager.mark_file_processed(file_name, source, content_hash)
        
        print(f"Processed {local_path} successfully")
        return True
        
    def get_similar_chunks(self, query: str, top_k: int = 3) -> List:
        """Retrieve similar chunks based on a query and return their text content."""
        index = VectorStoreIndex.from_vector_store(
            self.vector_store,
            storage_context=self.storage_context,
        )
        
        # Use retriever instead of query engine
        retriever = index.as_retriever(similarity_top_k=top_k)
        source_nodes = retriever.retrieve(query)
        
        # Extract text content from the source nodes and normalize Unicode characters
        chunk_texts = [unicodedata.normalize('NFKC', node.text) for node in source_nodes]
        
        return chunk_texts

if __name__ == "__main__":
    # Example usage
    processor = PDFProcessor()
    print(processor.get_similar_chunks("What is Bakra Beverage"))
    # Process a single PDF
    # processor.process_pdf("path/to/your/pdf")
    # Process a directory of PDFs
    # processor.process_directory("path/to/your/pdf/directory") 