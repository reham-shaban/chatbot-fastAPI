from fastapi import APIRouter, HTTPException, UploadFile, File
from groq import Groq
from typing import Dict
from dotenv import load_dotenv
import os, cohere
from models.models import ChatMessage

router = APIRouter()

load_dotenv(dotenv_path='..../variables/.env')
groq_api_key = os.getenv('GROQ_API_KEY')
# cohere_api_key = os.getenv('COHERE_API_KEY')

# routers
@router.post("/response")
async def generate_response(message : ChatMessage):
    return {
        "question" : message,
        "response" : "This is a temporally response for testing"
        }

@router.post("/audio-to-text")
async def audio_to_text(file: UploadFile = File(...)) -> Dict[str, str]:
    try:
        groqClient = Groq(api_key=groq_api_key)
        transcription = groqClient.audio.transcriptions.create(
            file=(file.filename, await file.read()),
            model="whisper-large-v3",
            response_format="verbose_json",
        )
        return {"text": transcription.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate-title")
async def generate_chat_title(message : ChatMessage):
    try:
        cohereClient = cohere.Client(api_key='pczcIAiOQLKPrJo3wRKrKlZyZpsqkw7lJiEhuJdA')
        # Sending the request to the chat model
        print("message: ", message.message)
        response = cohereClient.chat(
            model="command-r-plus",
            message=f"Generate a tilte for a chat with a cutomer service bot that starts with this message: {message.message}. Note that the tilte should be in the same languase of the message."
        )

        # Returning the response text
        return {"title": response.text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))