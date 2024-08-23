from fastapi import APIRouter, Request, Response, FastAPI
from http import HTTPStatus
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from telegram.ext._contexttypes import ContextTypes
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import os, uuid, logging
from services.rag_pipeline import RAGPipeline
from services.vectorstore_manager import DocumentsPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Load environment variables
dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../variables/.env'))
load_dotenv(dotenv_path=dotenv_path)
telegram_api_token = os.getenv('TELEGRAM_API_TOKEN')
embedding_model_name = os.getenv('EMBEDDING_MODEL_NAME')
hugging_api_key = os.getenv('HUGGING_FACE_API_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')
cohere_api_key = os.getenv('COHERE_API_KEY')
weaviate_cluster_URL = os.getenv('WEAVIATE_CLUSTER_URL')
weaviate_api_key = os.getenv('WEAVIATE_API_KEY')
weaviate_collection_name = os.getenv('WEAVIATE_COLLECTION_NAME')
app_url = os.getenv('APP_URL')

# Placeholder for the RAGPipeline instance
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

# Initialize python telegram bot
ptb = (
    Application.builder()
    .updater(None)
    .token(telegram_api_token)
    .read_timeout(7)
    .get_updates_read_timeout(42)
    .build()
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    await ptb.bot.setWebhook(f"{app_url}/telegram")  # Replace with your webhook URL
    async with ptb:
        await ptb.start()
        yield
        await ptb.stop()

@router.post("/")
async def process_update(request: Request):
    req = await request.json()
    update = Update.de_json(req, ptb.bot)
    await ptb.process_update(update)
    return Response(status_code=HTTPStatus.OK)

# Start command handler
async def start(update, _: ContextTypes.DEFAULT_TYPE):
    """Send a message and create new id when the command /start is issued."""
    # Generate a unique conversation_id
    global conversation_id 
    conversation_id = str(uuid.uuid4())
   
    await update.message.reply_text('مرحبًا بك في بوت خدمة العملاء لدينا. كيف يمكنني مساعدتك؟')

ptb.add_handler(CommandHandler("start", start))

# Message handler
async def handle_message(update, context):
    """Handle incoming messages."""
    user_input = update.message.text
    chat_id = update.message.chat_id
    
    # Generate a response using the conversation_id
    response = rag_pipeline.generate_response(user_input, conversation_id=conversation_id)
    
    # Send a response back to the user
    await context.bot.send_message(chat_id=chat_id, text=response)

# Register the message handler
ptb.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
