import os
from dotenv import load_dotenv
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.documents import html_to_json
from app.services.vectorstore import DocumentsPipeline
from app.models.models import MetadataFilter

# Initialize router
router = APIRouter()

# get env variables
load_dotenv(dotenv_path='..../variables/.env')
embedding_model_name = os.getenv('EMBEDDING_MODEL_NAME')
hugging_api_key = os.getenv('HUGGING_FACE_API_KEY')
weaviate_cluster_URL = os.getenv('WEAVIATE_CLUSTER_URL')
weaviate_api_key = os.getenv('WEAVIATE_API_KEY')
weaviate_collection_name = os.getenv('WEAVIATE_COLLECTION_NAME')

# Initialize the DocumentsPipeline instance
pipeline = DocumentsPipeline(
    embedding_model_name=embedding_model_name,
    hugging_api_key=hugging_api_key,
    collection_name=weaviate_collection_name,
    cluster_URL=weaviate_cluster_URL,
    weaviate_api_key=weaviate_api_key,
    text_key
)

# Routes
# convert html file to json
@router.post("/html-to-json")
async def convert_html(file: UploadFile = File(...)):
    try:
        # Read the HTML content from the uploaded file
        html_content = await file.read()

        # Convert HTML to JSON content
        json_content = html_to_json(html_content.decode("utf-8"))

        # Return the JSON content as a response
        return json_content

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# update env variables
@router.post("/update-env")
async def update_env_variables(embedding_model_name: str = None, hugging_api_key: str = None, weaviate_cluster_URL: str = None, weaviate_api_key: str = None, weaviate_collection_name: str = None):
    try:
        # Path to your .env file
        env_path = '.../variables/.env'

        # Read current .env file
        with open(env_path, 'r') as file:
            lines = file.readlines()

        # Create a dictionary of current env variables
        env_vars = {}
        for line in lines:
            key, value = line.strip().split('=', 1)
            env_vars[key] = value

        # Update the variables if new values are provided
        if embedding_model_name:
            env_vars['EMBEDDING_MODEL_NAME'] = embedding_model_name
        if hugging_api_key:
            env_vars['HUGGING_FACE_API_KEY'] = hugging_api_key
        if weaviate_cluster_URL:
            env_vars['WEAVIATE_CLUSTER_URL'] = weaviate_cluster_URL
        if weaviate_api_key:
            env_vars['WEAVIATE_API_KEY'] = weaviate_api_key
        if weaviate_collection_name:
            env_vars['WEAVIATE_COLLECTION_NAME'] = weaviate_collection_name

        # Write the updated variables back to the .env file
        with open(env_path, 'w') as file:
            for key, value in env_vars.items():
                file.write(f"{key}={value}\n")

        # Reload the .env file to update the environment variables in the running app
        load_dotenv(dotenv_path=env_path)

        # Update your pipeline or any other components that depend on these variables
        pipeline = DocumentsPipeline(
            embedding_model_name=os.getenv('EMBEDDING_MODEL_NAME'),
            hugging_api_key=os.getenv('HUGGING_FACE_API_KEY'),
            collection_name=os.getenv('WEAVIATE_COLLECTION_NAME'),
            cluster_URL=os.getenv('WEAVIATE_CLUSTER_URL'),
            weaviate_api_key=os.getenv('WEAVIATE_API_KEY')
        )

        return {"status": "Environment variables updated successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# delete document from collection
@router.delete("/delete-documents")
async def delete_documents(filter: MetadataFilter):
    response = pipeline.delete_documents_by_metadata(filter.property, filter.metadata_filter)
    
    if response:
        return {"status": "Documents deleted successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to delete documents")

# search document from collection
@router.get("/search-documents")
async def search_documents(filter: MetadataFilter):
    documents = pipeline.search_documents_by_metadata(filter.property, filter.metadata_filter)
    
    if documents is not None:
        return {"documents": documents}
    else:
        raise HTTPException(status_code=500, detail="Failed to search documents")

