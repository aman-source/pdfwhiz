from docx import Document

class DocProcessor:
    """Handles text extraction from DOCX files."""

    def extract_text(self, file):
        """Extracts text from a DOCX file."""
        try:
            doc = Document(file)
            text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
            return text.strip()
        except Exception as e:
            return f"Error extracting text from DOCX: {e}"
