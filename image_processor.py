import pytesseract
from PIL import Image

class ImageProcessor:
    """Handles text extraction from image files using Tesseract OCR."""

    def extract_text(self, image):
        """Extract text from an image file."""
        try:
            img = Image.open(image)
            text = pytesseract.image_to_string(img)
            return text.strip()
        except Exception as e:
            return f"Error extracting text from image: {e}"
