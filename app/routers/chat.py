from fastapi import APIRouter, HTTPException, Body
from models.models import ChatRequest

router = APIRouter()

@router.post("/response")
async def generate_response(message : ChatRequest):
    return {
        "question" : message,
        "response" : "This is a temporally response for testing"
        }