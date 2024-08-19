from fastapi import APIRouter, HTTPException, UploadFile, File
from groq import Groq
from typing import Dict
from dotenv import load_dotenv
import os, cohere
from models.models import ChatMessage
from app.services.vectorstore import DocumentsPipeline
from app.services.rag import RAGPipeline

router = APIRouter()

# get env variables
dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../variables/.env'))
load_dotenv(dotenv_path=dotenv_path)
embedding_model_name = os.getenv('EMBEDDING_MODEL_NAME')
hugging_api_key = os.getenv('HUGGING_FACE_API_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')
cohere_api_key = os.getenv('COHERE_API_KEY')
weaviate_cluster_URL = os.getenv('WEAVIATE_CLUSTER_URL')
weaviate_api_key = os.getenv('WEAVIATE_API_KEY')
weaviate_collection_name = os.getenv('WEAVIATE_COLLECTION_NAME')

# doc_pipeline = DocumentsPipeline(
#             collection_name=weaviate_collection_name,
#             embedding_model_name=embedding_model_name,
#             cluster_URL=weaviate_cluster_URL,
#             weaviate_api_key=weaviate_api_key,
#             hugging_api_key=hugging_api_key
#         )
# vectorstore = doc_pipeline.load_vector_store_from_collection()
# rag_pipeline = RAGPipeline(vectorstore=vectorstore, conversation_id=1, cohere_api_key=cohere_api_key)
# print("------rag------")

# routers
from fastapi.responses import StreamingResponse

co = cohere.Client(api_key='NO7yfaSUsE44j2uPSDbGQEcJpPmVAhIiWzAl3omw')

@router.get("/tell-joke")
async def tell_joke():
    async def joke_stream():
        response = co.chat_stream(
            model="command-r-plus",
            message="tell me a joke"
        )
        for event in response:
            if event.event_type == "text-generation":
                yield event.text
            
    return StreamingResponse(joke_stream(), media_type="text/plain")

# Run the app with `uvicorn filename:app --reload`

# @router.post("/get-response")
# async def get_response(question : str, conversation_id : int):
#     vectorstore = load_vector_store_from_collection()
#     cohere_api_key = 'pczcIAiOQLKPrJo3wRKrKlZyZpsqkw7lJiEhuJdA'
#     chat = RAGPipeline(vectorstore, cohere_api_key, k_number)
#     response = chat.generate_response(question, conversation_id)
#     return {"response" : response}

# @router.post("/response")
# async def generate_response(message : ChatMessage):
#     return {
#         "question" : message,
#         "response" : "This is a temporally response for testing"
#         }

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