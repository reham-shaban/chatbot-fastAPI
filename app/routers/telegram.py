from fastapi import APIRouter, BackgroundTasks, Request
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.constants import ChatAction
from services.rag_pipeline import RAGPipeline
from services.vectorstore_manager import DocumentsPipeline
from dotenv import load_dotenv
import os
import nest_asyncio
import logging
import asyncio

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

# Apply nest_asyncio to handle nested event loops
nest_asyncio.apply()

# Set up logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Global variables for bot and application
bot = None
application = None

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('مرحبًا بك في بوت خدمة العملاء لدينا. كيف يمكنني مساعدتك؟')

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text
    context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    response = rag_pipeline.generate_response(user_input)
    await update.message.reply_text(response)

# Background task to run the Telegram bot
async def run_bot():
    global bot, application
    application = Application.builder().token(telegram_api_token).build()
    bot = application.bot  # Initialize the bot

    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Start the bot
    await application.run_polling(allowed_updates=Update.ALL_TYPES)

@router.post("/webhook")
async def webhook(request: Request, background_tasks: BackgroundTasks):
    global bot, application
    data = await request.json()
    update = Update.de_json(data, bot)
    background_tasks.add_task(application.process_update, update)
    return {"status": "ok"}

@router.on_event("startup")
async def startup_event():
    asyncio.create_task(run_bot())
