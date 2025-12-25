from pathlib import Path
from typing import List, Dict
import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader


class PDFHandler:
    """
    PDF document handler.  
    It extracts texts from PDFs and store embeddings in chroma DB.
    
    """

    def __init__(self, pdf_dir: Path, db_dir: Path):
        self.pdf_dir = pdf_dir
        self.db_dir = db_dir

        self.chunk_size = 1000

        self.chroma_db = chromadb.PersistentClient(path=str(db_dir))
        self.collection = self.chroma_db.get_or_create_collection(
            name="test",
            metadata={
                "description": "test"
            }
        )
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """
        Given path of a PDF, extract the text from the PDF
        
        :param pdf_path: Path of a PDF
        :type pdf_path: Path
        :return: text extracted from PDF
        :rtype: str
        """

        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            
            return text
        except Exception as e:
            print("!!!!!--extract_text_from_pdf(),{e}")
            return None
        
    def chunk_text(self, text: str) -> List[str]:
        """
        Docstring for chunk_text
        
        :param self: Description
        :param text: Description
        :type text: str
        :return: Description
        :rtype: List[str]
        """
        chunks = []
        overlap = self.chunk_size // 4  # 25% overlap
        
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > self.chunk_size // 2:  # Only break if we're past halfway
                    chunk = chunk[:break_point + 1]
                    end = start + break_point + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
        
        return [c for c in chunks if len(c.strip()) > 50]  # Filter tiny chunks
        
    def index_pdf(self, pdf_path: Path) -> Dict[str, int]:
        """
        Docstring for index_pdf
        
        :param self: Description
        :param pdf_path: Description
        :type pdf_path: Path
        :return: Description
        :rtype: Dict[str, int]
        """

        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            return {
                "chunks": 0,
                "error": "No text extracted."
            }
        
        chunks = self.chunk_text(text)
        if not chunks:
            return {
                "chunks": 0,
                "error": "No chunk created"
            }
        
        embeddings = self.embedding_model.encode(chunks, show_progress_bar=False).tolist()



def main():
    pdf_dir = Path("./data/pdf")
    db_dir = Path("./data/db")
    
    handler = PDFHandler(pdf_dir, db_dir)
    print(handler.extract_text_from_pdf(Path("./data/pdf/OWASP-Top-10-for-Agentic-Applications-2026-12.6-1.pdf")))

    # embeddings = handler.embedding_model.encode(["This is a test."])

if __name__ == "__main__":
    main()