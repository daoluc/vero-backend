import os
import hashlib
from typing import List, Optional, Dict, Any
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
        
        # Create or get the collections
        self.embeddings_collection = chroma_client.get_or_create_collection("pdf_embeddings")
        self.metadata_collection = chroma_client.get_or_create_collection("pdf_metadata")
        
        # Create vector store
        self.vector_store = ChromaVectorStore(chroma_collection=self.embeddings_collection)
        
        # Create storage context
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        
        # Create service context
        self.service_context = ServiceContext.from_defaults()
        
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read the file in chunks to handle large files
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _get_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """Get metadata for a file."""
        file_name = os.path.basename(file_path)
        file_hash = self._calculate_file_hash(file_path)
        
        return {
            "file_path": file_path,
            "file_name": file_name,
            "content_hash": file_hash
        }
    
    def _check_duplicate(self, metadata: Dict[str, Any]) -> Optional[str]:
        """Check if a file with the same metadata exists and return its ID if found."""
        # Query the metadata collection for the file path
        results = self.metadata_collection.get(
            where={"file_path": metadata["file_path"]},
            include=["metadatas", "ids"]
        )
        
        if results and results["ids"]:
            return results["ids"][0]
        return None
    
    def _store_metadata(self, metadata: Dict[str, Any]) -> str:
        """Store file metadata and return the ID."""
        # Generate a unique ID for the metadata
        metadata_id = f"meta_{metadata['content_hash']}"
        
        # Store the metadata
        self.metadata_collection.add(
            ids=[metadata_id],
            metadatas=[metadata]
        )
        
        return metadata_id
    
    def _delete_old_embeddings(self, metadata_id: str) -> None:
        """Delete embeddings associated with the old version of a file."""
        # Get the metadata to find the content hash
        results = self.metadata_collection.get(
            ids=[metadata_id],
            include=["metadatas"]
        )
        
        if results and results["metadatas"]:
            old_hash = results["metadatas"][0]["content_hash"]
            
            # Delete embeddings with the old hash
            self.embeddings_collection.delete(
                where={"content_hash": old_hash}
            )
    
    def process_pdf(self, pdf_path: str) -> None:
        """Process a single PDF file and store its embeddings with deduplication."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        if not pdf_path.lower().endswith('.pdf'):
            raise ValueError(f"File is not a PDF: {pdf_path}")
        
        # Get file metadata
        metadata = self._get_file_metadata(pdf_path)
        
        # Check for duplicates
        existing_id = self._check_duplicate(metadata)
        
        if existing_id:
            # Get the existing metadata to compare content hashes
            existing_metadata = self.metadata_collection.get(
                ids=[existing_id],
                include=["metadatas"]
            )
            
            if existing_metadata and existing_metadata["metadatas"]:
                existing_hash = existing_metadata["metadatas"][0]["content_hash"]
                
                # If content hash is the same, the file is identical - skip processing
                if existing_hash == metadata["content_hash"]:
                    print(f"Skipping {pdf_path} - identical content already processed")
                    return
                
                # If content hash is different, delete old embeddings
                print(f"Content changed for {pdf_path} - reprocessing")
                self._delete_old_embeddings(existing_id)
        
        # Load the PDF
        documents = SimpleDirectoryReader(
            input_files=[pdf_path]
        ).load_data()
        
        # Add content hash to each document's metadata
        for doc in documents:
            doc.metadata["content_hash"] = metadata["content_hash"]
        
        # Create index and store embeddings
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=self.storage_context,
            service_context=self.service_context
        )
        
        # Store the metadata
        self._store_metadata(metadata)
        
        print(f"Processed {pdf_path} successfully")
        
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