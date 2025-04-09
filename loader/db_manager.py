import os
import hashlib
from typing import Optional
from sqlalchemy import create_engine, Column, String, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# Create the base class for declarative models
Base = declarative_base()

class ProcessedFile(Base):
    """Model to track processed PDF files and their content hashes."""
    __tablename__ = 'processed_files'
    
    file_path = Column(String, primary_key=True)
    folder_id = Column(String, nullable=False)
    content_hash = Column(String, nullable=False)
    processed_at = Column(DateTime, default=datetime.utcnow)
    is_processed = Column(Boolean, default=True)
    
    def __repr__(self):
        return f"<ProcessedFile(file_path='{self.file_path}', content_hash='{self.content_hash}')>"

class DatabaseManager:
    """Manager class for database operations related to processed files."""
    
    def __init__(self, db_path: str = "processed_files.db"):
        """Initialize the database manager with the specified database path."""
        self.db_path = db_path
        self.engine = create_engine(f"sqlite:///{db_path}")
        self.Session = sessionmaker(bind=self.engine)
        
        # Create tables if they don't exist
        Base.metadata.create_all(self.engine)
    
    def calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of a file's content."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read the file in chunks to handle large files efficiently
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        
        return sha256_hash.hexdigest()
    
    def is_file_processed(self, content_hash: str) -> bool:
        """Check if a file has already been processed based on its path or content hash."""
        session = self.Session()
        try:           
            hash_record = session.query(ProcessedFile).filter_by(content_hash=content_hash).first()
            if hash_record:
                return True
            
            return False
        finally:
            session.close()
    
    def mark_file_processed(self, file_path: str, folder_id: str, content_hash: Optional[str] = None) -> None:
        """Mark a file as processed in the database."""
        if content_hash is None:
            content_hash = self.calculate_file_hash(file_path)
        
        session = self.Session()
        try:
            # Check if file is already in the database
            existing_file = session.query(ProcessedFile).filter_by(file_path=file_path).first()
            
            if existing_file:
                # Update existing record
                existing_file.content_hash = content_hash
                existing_file.folder_id = folder_id
                existing_file.processed_at = datetime.utcnow()
                existing_file.is_processed = True
            else:
                # Create new record
                new_file = ProcessedFile(
                    file_path=file_path,
                    folder_id=folder_id,
                    content_hash=content_hash
                )
                session.add(new_file)
            
            session.commit()
        finally:
            session.close()
    
    def get_processed_files(self, folder_id: Optional[str] = None) -> list:
        """Get all processed files, optionally filtered by folder_id."""
        session = self.Session()
        try:
            query = session.query(ProcessedFile)
            if folder_id:
                query = query.filter_by(folder_id=folder_id)
            return query.all()
        finally:
            session.close() 