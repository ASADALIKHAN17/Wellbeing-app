from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, BackgroundTasks
from typing import List
from app.core.database import get_database
from app.routes.auth import get_current_user
from app.models.report import ReportResponse, ReportInDB, ExtractedData
from app.services.pdf_service import extract_text_from_pdf
from app.services.gemini_service import analyze_health_report
from bson import ObjectId
import shutil
import os
import re
from datetime import datetime, timedelta

router = APIRouter()

@router.get("/trends")
async def get_health_trends(
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    reports_cursor = db.reports.find({"user_id": str(current_user["_id"])}).sort("upload_date", 1)
    reports = await reports_cursor.to_list(length=100)
    
    trends = {}
    
    # 1. Build a map of normalized_key -> best_display_name
    # Iterate in reverse order (newest first) to prefer newer names for display
    param_name_map = {}
    for report in reversed(reports):
        data = report.get("extracted_data", {})
        for param in data:
            # Normalize: Remove parentheses, lowercase, strip, collapse spaces
            # "Hemoglobin (Hb)" -> "Hemoglobin" -> "hemoglobin"
            clean_param = re.sub(r'\(.*?\)', '', param)
            norm_key = " ".join(clean_param.strip().lower().split())
            
            if norm_key not in param_name_map:
                param_name_map[norm_key] = param.strip()
    
    for report in reports:
        date = report.get("upload_date").isoformat()
        data = report.get("extracted_data", {})
        
        for param, val in data.items():
            # Get the canonical display name for this parameter
            clean_param = re.sub(r'\(.*?\)', '', param)
            norm_key = " ".join(clean_param.strip().lower().split())
            
            # Use the canonical display name
            param_key = param_name_map.get(norm_key, param.strip())
            
            num_val = None
            unit = ""
            ref_range = None
            
            if isinstance(val, dict):
                num_val = val.get("value")
                unit = val.get("unit", "")
                ref_range = val.get("reference_range", None)
            elif isinstance(val, str):
                # Legacy support: extract number from string "14.5 g/dL"
                match = re.search(r"(\d+\.?\d*)", val)
                if match:
                    num_val = float(match.group(1))
                    unit = val.replace(match.group(1), "").strip()
            
            if num_val is not None:
                if param_key not in trends:
                    trends[param_key] = {
                        "unit": unit,
                        "reference_range": ref_range,
                        "data": []
                    }
                trends[param_key]["data"].append({
                    "date": date,
                    "value": num_val,
                    "referenceRange": ref_range
                })
                
    # Sort data points by date just in case
    for param in trends:
        trends[param]["data"].sort(key=lambda x: x["date"])
        
    return trends

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload", response_model=ReportResponse)
async def upload_report(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    # Save file locally
    file_path = f"{UPLOAD_DIR}/{current_user['_id']}_{datetime.now().timestamp()}_{file.filename}"
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    print(f"File saved to {file_path}")
        
    # Extract Text
    print("Extracting text from PDF...")
    text = await extract_text_from_pdf(content)
    print(f"Extracted {len(text)} characters.")
    if not text.strip():
        print("Text extraction failed or returned empty string.")
        raise HTTPException(status_code=400, detail="Could not extract text from the uploaded PDF. Please ensure it is a valid text-based PDF report.")
    
    # Analyze with Gemini (pass bytes for native visual/layout analysis)
    print("Calling Gemini for native PDF analysis...")
    gemini_result = await analyze_health_report(pdf_bytes=content)
    print("Gemini analysis complete.")
    
    print("Preparing report for DB...")
    try:
        report_in_db = ReportInDB(
            user_id=str(current_user["_id"]),
            extracted_data=gemini_result.get("extracted_data", {}),
            gemini_analysis=gemini_result.get("analysis"),
            pdf_path=file_path,
            upload_date=datetime.utcnow()
        )
    except Exception as e:
        print(f"FAILED to create ReportInDB model: {e}")
        raise HTTPException(status_code=500, detail=f"Data validation error: {str(e)}")
    
    print("Inserting into MongoDB...")
    new_report = await db.reports.insert_one(report_in_db.dict(by_alias=True, exclude={"id"}))
    created_report = await db.reports.find_one({"_id": new_report.inserted_id})
    print("Report saved to DB.")
    
    # Update user's last report date and next checkup
    next_checkup = datetime.utcnow() + timedelta(days=90)
    await db.users.update_one(
        {"_id": ObjectId(current_user["_id"])},
        {"$set": {
            "last_report_date": datetime.utcnow(),
            "next_checkup_date": next_checkup
        }}
    )
    
    created_report["_id"] = str(created_report["_id"])
    return created_report

@router.get("/", response_model=List[ReportResponse])
async def list_reports(
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    reports_cursor = db.reports.find({"user_id": str(current_user["_id"])}).sort("upload_date", -1)
    reports = await reports_cursor.to_list(length=100)
    # Convert ObjectId to str
    for report in reports:
        report["_id"] = str(report["_id"])
    return reports

@router.get("/{report_id}", response_model=ReportResponse)
async def get_report(
    report_id: str,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    if not ObjectId.is_valid(report_id):
        raise HTTPException(status_code=400, detail="Invalid ID")
        
    report = await db.reports.find_one({
        "_id": ObjectId(report_id),
        "user_id": str(current_user["_id"])
    })
    
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
        
    report["_id"] = str(report["_id"])
    return report

@router.delete("/{report_id}")
async def delete_report(
    report_id: str,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    if not ObjectId.is_valid(report_id):
        raise HTTPException(status_code=400, detail="Invalid ID")
        
    result = await db.reports.delete_one({
        "_id": ObjectId(report_id),
        "user_id": str(current_user["_id"])
    })
    
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Report not found")
        
    return {"status": "success"}
