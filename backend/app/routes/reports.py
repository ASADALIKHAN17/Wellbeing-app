from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, BackgroundTasks
from typing import List
from app.core.database import get_database
from app.routes.auth import get_current_user
from app.models.report import ReportResponse, ReportInDB, ExtractedData
from app.services.pdf_service import extract_text_from_pdf, parse_health_parameters
from app.services.gemini_service import analyze_health_report
from bson import ObjectId
import shutil
import os
from datetime import datetime, timedelta

router = APIRouter()

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
        
    # Extract Text
    text = await extract_text_from_pdf(content)
    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from the uploaded PDF. Please ensure it is a valid text-based PDF report.")
    
    # Parse basic params locally
    parsed_data_dict = await parse_health_parameters(text)
    
    # Analyze with Gemini
    # Note: For strict 2-day timeline, we do this synchronously here or strictly await it.
    # ideally this should be a background task, but user wants immediate results in the flow described.
    gemini_analysis = await analyze_health_report(text, parsed_data_dict)
    
    report_in_db = ReportInDB(
        user_id=str(current_user["_id"]),
        extracted_data=parsed_data_dict,
        gemini_analysis=gemini_analysis,
        pdf_path=file_path,
        upload_date=datetime.utcnow()
    )
    
    new_report = await db.reports.insert_one(report_in_db.dict(by_alias=True, exclude={"id"}))
    created_report = await db.reports.find_one({"_id": new_report.inserted_id})
    
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
