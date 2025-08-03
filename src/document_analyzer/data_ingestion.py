import os
import fitz
import uuid
import datetime
from datetime import datetime as dt
from pathlib import Path
from io import BytesIO
from requests import session
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException, ExceptionSeverity

class DocumentHandler:
    """
    Handles PDF saving and reading operations.
    Automatically logs all actions and supports session-based organization.

    """
    def __init__(self,data_dir= None, session_id = None):
        try:
            self.log=CustomLogger().get_logger(__name__)
            self.data_dir = data_dir or os.getenv(
                "DATA_STORAGE_PATH", os.path.join(os.getcwd(), "data", "document_analysis")
            )
            self.session_id = session_id or f"session_{dt.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            self.session_path = os.path.join(self.data_dir, self.session_id)
            os.makedirs(self.session_path, exist_ok=True) 

            self.log.info("PDFHandler initialized", session_id= self.session_id, session_path=self.session_path)
        except Exception as e:
            #self.log.error(f"Error initializing DocumentHandler: {e}")
            raise DocumentPortalException("Error initializing DocumentHandler", e, ExceptionSeverity.HIGH)
    
    def save_pdf(self, uploaded_file):
        try:
            filename = os.path.basename(uploaded_file.name)

            if not filename.lower().endswith(".pdf"):
                raise DocumentPortalException("Invalid File Type. Only PDFs are allowed.")
            
            save_path= os.path.join(self.session_path, filename)

            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            self.log.info("PDF Saved successfully" , file=filename, save_path=save_path, session_id=self.session_id)
            return save_path
        except Exception as e:
            #self.log.error(f"Error saving Document: {e}")
            raise DocumentPortalException("Error saving Document", e)
    
    def read_pdf(self,pdf_path:str):
        try:
            text_chunks = []
            with fitz.open(pdf_path) as doc:
                for page_num, page in enumerate (doc, start=1): # type: ignore
                    text_chunks.append(f"\n--- Page {page_num} ---\n{page.get_text()}")
            text = "\n".join(text_chunks)
            self.log.info("PDF Read successfully", pdf_path=pdf_path,session_id=self.session_id,pages=len(text_chunks))
            return text
        except Exception as e:
            #self.log.error(f"Error reading Document: {e}")
            raise DocumentPortalException("Error reading Document", e)


