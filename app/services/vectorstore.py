import os
import weaviate
from dotenv import load_dotenv
from weaviate.classes.init import Auth
from weaviate.classes.query import Filter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_weaviate.vectorstores import WeaviateVectorStore

# pipeline
class DocumentsPipeline :
    def __init__(self, embedding_model_name, hugging_api_key, collection_name, cluster_URL, weaviate_api_key, text_key):
        self.collection_name = collection_name
        self.cluster_URL = cluster_URL
        self.weaviate_api_key = weaviate_api_key
        self.text_key = text_key
        self.client = self._init_weaviate_connection()
        self.embedder = self._init_embedding_model(hugging_api_key, embedding_model_name)
        
    def _init_embedding_model(self, api_key, model_name):
        embedder = HuggingFaceInferenceAPIEmbeddings(
            api_key=api_key, model_name=model_name
        )
        return embedder
    
    def _init_weaviate_connection(self):
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=self.cluster_URL,
            auth_credentials=Auth.api_key(self.weaviate_api_key),
        )
        return client 
    
    def _load_vector_store_from_collection(self):     
        vector_store = WeaviateVectorStore(
            client=self.client,
            index_name=self.collection_name ,
            text_key=self.text_key ,
            embedding=self.embedder
        )
        return vector_store
      
    def _get_collection(self):
        collection = self.client.collections.get(self.collection_name)
        return collection 
       
    def add_documents_data(self, my_documents):
        try:
            vector_store = self._load_vector_store_from_collection()
            vector_store.add_documents(documents=my_documents)
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
    
    def delete_documents_by_metadata(self , metadata_filter , property):
        """
        property : name (str) | active (bool) | date (str) 
        metadata_filter : the value of the property
        """
        collection = self._get_collection()
        search_filter = Filter.by_property(property).like(metadata_filter)
        return collection.data.delete_many(where=search_filter)
    
    def search_documents_by_metadata(self , metadata_filter , property):
        """
        property : name (str) | active (bool) | date (str) 
        metadata_filter : the value of the property
        """
        collection = self._get_collection()
        search_filter = Filter.by_property(property).like(metadata_filter)
        result = collection.query.fetch_objects(filters= search_filter)
        chunks = []
        for o in result.objects:
            chunks.extend(o.properties)
            
        return chunks
