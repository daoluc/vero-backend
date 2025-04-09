import os
from typing import List, Optional, Callable
from pathlib import Path
from dotenv import load_dotenv
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
import pickle

load_dotenv()

class GoogleDriveLoader:
    def __init__(self, credentials_path: str = "./loader/credentials.json", scopes: List[str] = None):
        """Initialize the Google Drive loader with service account credentials."""
        self.credentials_path = credentials_path
        self.scopes = scopes or ['https://www.googleapis.com/auth/drive.readonly']
        self.service = self._get_drive_service()
        
    def _get_drive_service(self):
        """Set up and return Google Drive service using service account credentials."""
        credentials = service_account.Credentials.from_service_account_file(
            self.credentials_path, 
            scopes=self.scopes
        )
        return build('drive', 'v3', credentials=credentials)
    
    def list_pdfs_in_folder(self, folder_id: str) -> List[dict]:
        """List all PDF files in the specified folder."""
        results = self.service.files().list(
            q=f"'{folder_id}' in parents and mimeType='application/pdf'",
            fields="files(id, name)"
        ).execute()
        
        return results.get('files', [])
    
    def download_file(self, file_id: str, output_path: str) -> str:
        """Download a file from Google Drive and return the local path."""
        request = self.service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            
        # Save the file
        fh.seek(0)
        with open(output_path, 'wb') as f:
            f.write(fh.read())
            
        return output_path
    
    def process_folder(self, folder_id: str, output_dir: str, file_processor: Callable[[str], None]) -> None:
        """Download files from a folder, process them using the provided processor, and clean up."""
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Get list of PDFs
            pdfs = self.list_pdfs_in_folder(folder_id)
            
            # Process each PDF
            for pdf in pdfs:
                output_path = os.path.join(output_dir, pdf['name'])
                try:
                    # Download the file
                    local_path = self.download_file(pdf['id'], output_path)
                    # Process the file
                    file_processor(local_path)
                finally:
                    # Always delete the file after processing, even if processing fails
                    if os.path.exists(output_path):
                        os.remove(output_path)
                
        finally:
            # Clean up the output directory
            if os.path.exists(output_dir):
                if not os.listdir(output_dir):
                    os.rmdir(output_dir)

if __name__ == "__main__":
    # Example usage
    from pdf_processor import PDFProcessor
    
    # Initialize the loader and processor
    # Note: credentials.json should now be a service account key file
    loader = GoogleDriveLoader()
    processor = PDFProcessor()
    
    # Process PDFs from the specified Google Drive folder
    folder_id = "16QhQl_DD79M_2UEXIwPvFQKikwqZsyda"
    output_dir = "downloaded_pdfs"
    
    # Use the process_pdf method from PDFProcessor as the callback
    loader.process_folder(folder_id, output_dir, processor.process_pdf)