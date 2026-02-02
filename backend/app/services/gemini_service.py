import os
from google import genai
from google.genai import types
import json
from app.core.config import get_settings
from app.models.report import GeminiAnalysis, ExtractedData

settings = get_settings()

# Initialize Gemini Client
client = genai.Client(api_key=settings.GEMINI_API_KEY)
model_id = "models/gemini-flash-latest" # Stable alias for Gemini 1.5 Flash


async def analyze_health_report(extracted_text: str = None, pdf_bytes: bytes = None) -> dict:
    """
    Analyzes health report using Gemini API.
    Can accept extracted text OR raw PDF bytes for native visual analysis.
    """
    
    # Base prompt focusing on medical logic
    base_prompt = """
    You are a professional health advisor. Analyze the following medical lab report.
    
    TASKS:
    1. EXTRACT DATA: Find all health parameters (like Hemoglobin, Sugar, Cholesterol, etc.). 
       For each parameter, you MUST extract:
       - The actual numerical value (as a floating point number).
       - The unit of measurement.
       - The reference range (MANDATORY - see instructions below).
       
    2. REFERENCE RANGE EXTRACTION (CRITICAL):
       For EVERY parameter you extract, you MUST provide a reference_range field.
       - FIRST: Look for the 'Reference Range', 'Normal Range', or 'Biological Reference Interval' in the report document itself.
       - IF NOT FOUND in document: Use standard Indian medical reference ranges:
         * Hemoglobin: "13-17" for men, "12-15" for women (use "12-17" if gender unknown)
         * Blood Sugar (Fasting): "70-100"
         * Blood Sugar (Random): "70-140"
         * HbA1c: "4-5.6"
         * Total Cholesterol: "125-200"
         * LDL Cholesterol: "0-100"
         * HDL Cholesterol: "40-60"
         * Triglycerides: "0-150"
         * Vitamin D: "30-100"
         * Vitamin B12: "200-900"
         * TSH: "0.5-5"
         * Creatinine: "0.6-1.2"
         * Uric Acid: "3.5-7.2"
         * SGPT/ALT: "0-40"
         * SGOT/AST: "0-40"
       - For any other parameter not listed above, use your medical knowledge to provide appropriate standard ranges.
       - IMPORTANT: The reference_range should be a string in format "min-max" (e.g., "13-17"), without units.
       
    3. ANALYZE: Compare values against the reference ranges and identify abnormalities.
    
    4. RECOMMEND: Provide dietary and lifestyle advice based on the results.

    OUTPUT FORMAT:
    You MUST return a JSON object with this EXACT structure:
    {
        "extracted_data": {
            "parameter_name": {
                "value": 12.5,
                "unit": "g/dL",
                "reference_range": "13-17"
            },
            ...
        },
        "analysis": {
            "summary": "brief overview of health status",
            "health_score": 85, 
            "abnormal_parameters": ["param1", "param2"],
            "dietary_suggestions": ["suggestion1", "suggestion2"],
            "foods_to_include": ["food1", "food2"],
            "foods_to_avoid": ["food1", "food2"],
            "lifestyle_tips": ["tip1", "tip2"],
            "doctor_consultation": true/false
        }
    }
    """

    if pdf_bytes:
        # Use native PDF processing for better layout/table understanding
        contents = [
            base_prompt,
            types.Part.from_bytes(data=pdf_bytes, mime_type='application/pdf')
        ]
    else:
        # Fallback to text if no bytes provided
        contents = f"{base_prompt}\n\nREPORT TEXT:\n{extracted_text}"
    
    try:
        response = client.models.generate_content(
            model=model_id,
            contents=contents,
            config={
                'response_mime_type': 'application/json',
            }
        )
    except Exception as e:
        print(f"Gemini API Call failed: {e}")
        return {
            "extracted_data": {},
            "analysis": {
                "summary": "Error: AI analysis service is currently unavailable.",
                "health_score": 0,
                "abnormal_parameters": [],
                "dietary_suggestions": [],
                "foods_to_include": [],
                "foods_to_avoid": [],
                "lifestyle_tips": [],
                "doctor_consultation": True
            }
        }
    
    # The new SDK with response_mime_type returns valid JSON in response.text
    try:
        data = json.loads(response.text)
        
        # Validate the analysis part using our Pydantic model
        if "analysis" in data:
            print("Validating analysis with GeminiAnalysis model...")
            # This ensures the AI output matches our expected schema for storage
            data["analysis"] = GeminiAnalysis(**data["analysis"]).dict()
            print("Analysis validation successful.")
            
        if "extracted_data" in data:
            print("Validating extracted_data...")
            # Ensure it's a dict
            if not isinstance(data["extracted_data"], dict):
                print(f"Warning: extracted_data is not a dict, it is {type(data['extracted_data'])}")
                data["extracted_data"] = {}
            else:
                # Use our model to ensure consistency
                validated_data = ExtractedData(data=data["extracted_data"])
                data["extracted_data"] = validated_data.dict().get("data", {})
                print("Extracted data validation successful.")
        return data
    except Exception as e:
        print(f"Error parsing/validating Gemini response: {e}")
        # Fallback to a valid structure that matches GeminiAnalysis model
        return {
            "extracted_data": {},
            "analysis": {
                "summary": "Error analyzing report content. The AI output was not in the expected format.",
                "health_score": 0,
                "abnormal_parameters": [],
                "dietary_suggestions": [],
                "foods_to_include": [],
                "foods_to_avoid": [],
                "lifestyle_tips": [],
                "doctor_consultation": True
            }
        }

print("Gemini key loaded:", bool(settings.GEMINI_API_KEY))
