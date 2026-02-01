import fitz  # PyMuPDF
import re
import logging

logger = logging.getLogger(__name__)

async def extract_text_from_pdf(file_content: bytes) -> str:
    """
    Extracts text from PDF bytes using PyMuPDF.
    """
    try:
        doc = fitz.open(stream=file_content, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        return ""

async def parse_health_parameters(text: str) -> dict:
    """
    Basic regex parsing for common parameters.
    This is a heuristic approach and might need refinement for specific report formats.
    """
    data = {}
    
    # Example Regex patterns (very basic, can be improved)
    patterns = {
        "hemoglobin": r"Hemoglobin\s*[:\-]?\s*([\d\.]+)",
        "blood_sugar_fasting": r"Fasting Blood Sugar\s*[:\-]?\s*([\d\.]+)",
        "total_cholesterol": r"Total Cholesterol\s*[:\-]?\s*([\d\.]+)",
        "tsh": r"TSH\s*[:\-]?\s*([\d\.]+)",
        "vitamin_d": r"Vitamin D\s*[:\-]?\s*([\d\.]+)"
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                data[key] = float(match.group(1))
            except ValueError:
                pass
                
    # Add full text for Gemini to analyze if regex fails or for more context
    # Note: We are not storing this in DB as per user request, but returning it for the service to use
    return data
