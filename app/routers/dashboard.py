import os
from dotenv import load_dotenv
from fastapi import APIRouter, UploadFile, File, HTTPException
from tempfile import NamedTemporaryFile
from app.services.vectorstore_manager import DocumentsPipeline
from app.services.convert_html_pipeline import ConvertHTMLPipeline
from models.models import Metadata

# Initialize router
router = APIRouter()

# get env variables
dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../variables/.env'))
load_dotenv(dotenv_path=dotenv_path)
embedding_model_name = os.getenv('EMBEDDING_MODEL_NAME')
hugging_api_key = os.getenv('HUGGING_FACE_API_KEY')
weaviate_cluster_URL = os.getenv('WEAVIATE_CLUSTER_URL')
weaviate_api_key = os.getenv('WEAVIATE_API_KEY')
weaviate_collection_name = os.getenv('WEAVIATE_COLLECTION_NAME')

# Routes
# Add document data from an HTML filer
@router.post("/add-document/")
async def add_document(name: str, active: bool, date: str, file: UploadFile = File(...)):
    """
    Adds document data from an uploaded HTML file to the Weaviate vector store.

    Args:
        file (UploadFile): The HTML file uploaded by the user.
        metadata (Metadata): Metadata associated with the document.

    Returns:
        dict: A response indicating whether the document was successfully added.
    """
    try:
        metadata = {
            "name":name,
            "active":active,
            "data":date
        }

        # Initialize DocumentsPipeline
        pipeline = DocumentsPipeline(
            collection_name=weaviate_collection_name,
            embedding_model_name=embedding_model_name,
            cluster_URL=weaviate_cluster_URL,
            weaviate_api_key=weaviate_api_key,
            hugging_api_key=hugging_api_key
        )

        # Create a temporary file to save the uploaded HTML file
        with NamedTemporaryFile(delete=False, suffix=".html") as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name

        print("metadata: ", metadata)
        # Add document data to the vector store
        success = pipeline.add_documents_data(html_path=temp_file_path, metadata=metadata)

        # Clean up the temporary HTML file
        os.remove(temp_file_path)

        if success:
            return {"status": "success", "message": "Document data added successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to add document data")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# Search document by metadata
@router.post("/search-document")
async def search_documents_by_metadata(property, metadata_filter):
    """
    Search documents by metadata.

    Args:
        property (str): The property to filter by (e.g., 'name', 'active', 'date').
        metadata_filter (str): The value to filter by.

    Returns:
        List[Dict[str, Any]]: A list of documents matching the filter criteria.
    """
    try:
        # Initialize DocumentsPipeline
        pipeline = DocumentsPipeline(
            collection_name=weaviate_collection_name,
            embedding_model_name=embedding_model_name,
            cluster_URL=weaviate_cluster_URL,
            weaviate_api_key=weaviate_api_key,
            hugging_api_key=hugging_api_key
        )

        # Search for documents using the specified property and filter
        chunks = pipeline.search_documents_by_metadata(property=property, metadata_filter=metadata_filter)

        return chunks

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# Delete document by metadata
@router.post("/delete-document")
async def delete_documents_by_metadata(property, metadata_filter):
    """
    Delete documents by metadata.

    Args:
        property (str): The property to filter by (e.g., 'name', 'active', 'date').
        metadata_filter (str): The value to filter by.

    Returns:
        List[Dict[str, Any]]: A list of documents matching the filter criteria.
    """
    try:
        # Initialize DocumentsPipeline
        pipeline = DocumentsPipeline(
            collection_name=weaviate_collection_name,
            embedding_model_name=embedding_model_name,
            cluster_URL=weaviate_cluster_URL,
            weaviate_api_key=weaviate_api_key,
            hugging_api_key=hugging_api_key
        )

        # Delete for documents using the specified property and filter
        chunks = pipeline.delete_documents_by_metadata(property=property, metadata_filter=metadata_filter)

        return chunks

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# Get all files info from collection
@router.get("/get-all-files")
async def get_all_files_unique_by_name():
    """
    Get all files info from collection.

    Returns:
        A list of dict that contains (name, active, date).
    """
    try:
        # Initialize DocumentsPipeline
        pipeline = DocumentsPipeline(
            collection_name=weaviate_collection_name,
            embedding_model_name=embedding_model_name,
            cluster_URL=weaviate_cluster_URL,
            weaviate_api_key=weaviate_api_key,
            hugging_api_key=hugging_api_key
        )

        # Fetch all unique files by name
        files = pipeline.get_all_files_uniqe_by_name()

        return files

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# update env variables
@router.post("/update-env")
async def update_env_variables(
    embedding_model_name: str = None, 
    hugging_api_key: str = None, 
    weaviate_cluster_URL: str = None, 
    weaviate_api_key: str = None, 
    weaviate_collection_name: str = None,
    groq_api_key: str = None,
    cohere_api_key: str = None
    ):
    try:
        # Path to your .env file
        env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../variables/.env'))

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
        if groq_api_key:
            env_vars['GROQ_API_KEY'] = groq_api_key
        if cohere_api_key:
            env_vars['COHERE_API_KEY'] = cohere_api_key

        # Write the updated variables back to the .env file
        with open(env_path, 'w') as file:
            for key, value in env_vars.items():
                file.write(f"{key}={value}\n")

        # Reload the .env file to update the environment variables in the running app
        load_dotenv(dotenv_path=env_path)

        return {"status": "Environment variables updated successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/update-template")
async def update_prompt_template(prompt_template: str):
    try:
        # Validate the template content (if necessary)
        if len(prompt_template.strip()) == 0:
            raise HTTPException(status_code=400, detail="Template content cannot be empty")

        # Path to your configuration file
        config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../config/prompt_template.txt'))

        # Write the new template to the configuration file
        with open(config_path, 'w', encoding='utf-8') as file:
            file.write(prompt_template)

        return {"status": "Prompt template updated successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
