import os
import google.generativeai as genai
import json
from app.core.config import get_settings
from app.models.report import GeminiAnalysis

settings = get_settings()

# Configure Gemini
genai.configure(api_key=settings.GEMINI_API_KEY)
model = genai.GenerativeModel("models/text-bison-001")


async def analyze_health_report(extracted_text: str, extracted_data: dict) -> dict:
    """
    Analyzes health report text using Gemini API and returns a structured JSON response.
    """
    
    prompt = f"""
    You are a health advisor analyzing a patient's lab report.
    
    Extracted Data:
    {json.dumps(extracted_data, indent=2)}
    
    Full Report Text (Context):
    {extracted_text[:5000]}  # Truncate to avoid token limits if necessary, though 1.5 has large context
    
    Normal Ranges (Indian standards):
    - Hemoglobin: 13-17 g/dL (men), 12-15 g/dL (women)
    - Blood Sugar (Fasting): 70-100 mg/dL
    - Total Cholesterol: < 200 mg/dL
    - Vitamin D: 30-100 ng/mL
    
    Tasks:
    1. Identify parameters outside normal range
    2. Assess overall health status
    3. Provide dietary recommendations using common Indian foods
    4. Suggest lifestyle modifications
    5. Indicate if doctor consultation is recommended
    
    Respond in STRICT JSON format matching this schema:
    {{
        "summary": "brief overview",
        "abnormal_parameters": ["param1", "param2"],
        "dietary_suggestions": ["suggestion1", "suggestion2"],
        "foods_to_include": ["food1", "food2"],
        "foods_to_avoid": ["food1", "food2"],
        "lifestyle_tips": ["tip1", "tip2"],
        "doctor_consultation": true/false
    }}
    
    IMPORTANT: Return ONLY the JSON string. Do not use markdown formatting like ```json or ```.
    """
    
    response = model.generate_content(prompt)
    response_text = response.text
    
    # robustly extract JSON from the response
    import re
    # Look for the first opening brace and the last closing brace
    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    
    if json_match:
        json_str = json_match.group(0)
    else:
        # If no braces found, use the whole text (likely will fail JSON load if not valid)
        json_str = response_text
        
    data = json.loads(json_str)
    
    # Return dict as requested, Pydantic upstream should handle it if needed, or if modifying signature
    return data

print("Gemini key loaded:", bool(os.getenv("GEMINI_API_KEY")))
