from PyPDF2 import PdfReader

class PDFProcessor:
    """Handles text extraction from PDF files."""

    def extract_text(self, file):
        """Extract text from a PDF file."""
        try:
            reader = PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text.strip()
        except Exception as e:
            return f"Error extracting text from PDF: {e}"
