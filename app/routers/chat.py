from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from groq import Groq
from typing import Dict, AsyncGenerator
from dotenv import load_dotenv
import os, cohere
from models.models import ChatMessage
from services.vectorstore_manager import DocumentsPipeline
from services.rag_pipeline import RAGPipeline

router = APIRouter()

# Load environment variables
dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../variables/.env'))
load_dotenv(dotenv_path=dotenv_path)
embedding_model_name = os.getenv('EMBEDDING_MODEL_NAME')
hugging_api_key = os.getenv('HUGGING_FACE_API_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')
cohere_api_key = os.getenv('COHERE_API_KEY')
weaviate_cluster_URL = os.getenv('WEAVIATE_CLUSTER_URL')
weaviate_api_key = os.getenv('WEAVIATE_API_KEY')
weaviate_collection_name = os.getenv('WEAVIATE_COLLECTION_NAME')

# routes
@router.post("/test-response")
async def get_response(question: str = Form(...), conversation_id: str = Form(...)):
    print(question)
    print(conversation_id)
    return {"response": "This is a test response"}

@router.post("/get-response")
async def get_response(question: str = Form(...), conversation_id: str = Form(...), is_en: bool = Form(...)):
    try:
        # Initialize document pipeline and RAGPipeline (asynchronous if possible)
        document_pipeline = DocumentsPipeline(
            collection_name=weaviate_collection_name,
            embedding_model_name=embedding_model_name,
            cluster_URL=weaviate_cluster_URL,
            weaviate_api_key=weaviate_api_key,
            hugging_api_key=hugging_api_key
        )
        collection = document_pipeline.get_collection()
        embedder = document_pipeline.init_embedding_model()

        chat = RAGPipeline(
            collection=collection,
            embedder=embedder,
            cohere_api_key=cohere_api_key,
        )
        
        response = chat.generate_response(question=question, conversation_id=conversation_id, is_en=is_en)
        return {"response": response}   
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream-response")
async def stream_response(question: str = Form(...), conversation_id: str = Form(...), is_en: bool = Form(...)):
    # Initialize document pipeline and RAGPipeline
    document_pipeline = DocumentsPipeline(
        collection_name=weaviate_collection_name,
        embedding_model_name=embedding_model_name,
        cluster_URL=weaviate_cluster_URL,
        weaviate_api_key=weaviate_api_key,
        hugging_api_key=hugging_api_key
    )
    collection = document_pipeline.get_collection()
    embedder = document_pipeline.init_embedding_model()

    rag_pipeline = RAGPipeline(
        collection=collection,
        embedder=embedder,
        cohere_api_key=cohere_api_key,
    )

    async def event_generator():
        try:
            # Stream the response from the pipeline
            async for chunk in rag_pipeline.stream_response(question, conversation_id=conversation_id, is_en=is_en):
                yield chunk
        except Exception as e:
            yield f"Error: {str(e)}"

    return StreamingResponse(event_generator(), media_type="text/plain")


@router.get("/stream-response-test")
async def tell_joke():
    async def joke_stream():
        co = cohere.Client(api_key=cohere_api_key)
        response = co.chat_stream(
            model="command-r-plus",
            message="tell me a joke"
        )
        for event in response:
            if event.event_type == "text-generation":
                yield event.text
            
    return StreamingResponse(joke_stream(), media_type="text/plain")

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