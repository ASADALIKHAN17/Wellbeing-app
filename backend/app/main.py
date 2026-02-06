from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import get_settings
from app.core.database import connect_to_mongo, close_mongo_connection
from app.routes import auth, users, reports, bmi

settings = get_settings()

app = FastAPI(title=settings.PROJECT_NAME)

# CORS Configuration
origins = ["*"]  # Allow all origins for production deployment

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database Events
app.add_event_handler("startup", connect_to_mongo)
app.add_event_handler("shutdown", close_mongo_connection)

# Routes
app.include_router(auth.router, prefix=f"{settings.API_V1_STR}/auth", tags=["auth"])
app.include_router(users.router, prefix=f"{settings.API_V1_STR}/users", tags=["users"])
app.include_router(reports.router, prefix=f"{settings.API_V1_STR}/reports", tags=["reports"])
app.include_router(bmi.router, prefix=f"{settings.API_V1_STR}/bmi", tags=["bmi"])

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
