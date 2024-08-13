from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from dotenv import load_dotenv
import os
import weaviate
from weaviate.classes.init import Auth
from langchain_weaviate.vectorstores import WeaviateVectorStore

load_dotenv(dotenv_path='..../variables/.env')
hugging_api_key = os.getenv('HUGGING_FACE_API_KEY')
weaviate_cluster_URL = os.getenv('WEAVIATE_CLUSTER_URL')
weaviate_api_key = os.getenv('WEAVIATE_API_KEY')
weaviate_collection_name = os.getenv('WEAVIATE_COLLECTION_NAME')
weaviate_text_key = ['text' , 'name' , 'date' , 'active']

# methods
def init_embedding_model(model_name):
    embedder = HuggingFaceInferenceAPIEmbeddings(
        api_key=hugging_api_key, model_name=model_name)
    return embedder

def init_weaviate_connection(cluster_url , api_key):
    client = weaviate.connect_to_weaviate_cloud(
    cluster_url=cluster_url,
    auth_credentials=Auth.api_key(api_key),
        )
    return client

def create_vector_store_weaviate(documents , embedder , client):
    vector_store =  WeaviateVectorStore.from_documents(documents=documents, embedding=embedder, client=client )
    return vector_store

def load_vector_store_from_collection() :
    client = init_weaviate_connection(weaviate_cluster_URL ,weaviate_api_key )
    vector_store = WeaviateVectorStore(
    client=client,
    index_name=weaviate_collection_name ,
    text_key=weaviate_text_key)
    return vector_store

def add_documents_data(documents ,text_key , collection_name , client ):
    try:
        vector_store = load_vector_store_from_collection(text_key, collection_name, client)
        vector_store.add_documents(documents=documents)
        return True  # Indicating success
    except ValueError as e:
        print(f"ValueError occurred: {e}")
        # This could happen if the documents are not in the expected format
    except ConnectionError as e:
        print(f"ConnectionError occurred: {e}")
        # This could happen if there's an issue connecting to the client
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        # This catches any other unforeseen errors
    finally:
        print("Document addition process completed.")
    
    return False  # Indicating failure if any exception was caught

