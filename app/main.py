import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from fastapi import FastAPI
from app.routers import chat, dashboard, telegram

app = FastAPI()

# Include the routers
app.include_router(chat.router, prefix="/chat")
app.include_router(dashboard.router, prefix="/dashboard")
app.include_router(telegram.router, prefix="/telegram")

@app.get("/")
async def root():
    return {"message": "Welcome to the FastAPI application!"}
