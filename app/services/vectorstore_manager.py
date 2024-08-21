import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.query import Filter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_weaviate.vectorstores import WeaviateVectorStore
from .convert_html_pipeline import ConvertHTMLPipeline


class DocumentsPipeline :
    def __init__(self, collection_name, embedding_model_name, cluster_URL, weaviate_api_key, hugging_api_key):
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self.cluster_URL = cluster_URL
        self.weaviate_api_key = weaviate_api_key
        self.text_key = 'text'
        self.hugging_api_key = hugging_api_key
        self.client = self._init_weaviate_connection()
        self.embedder = self.init_embedding_model()

    def init_embedding_model(self):
        embedder = HuggingFaceInferenceAPIEmbeddings(
        api_key=self.hugging_api_key, model_name=self.embedding_model_name)
        return embedder
    
    def _init_weaviate_connection(self):
        client = weaviate.connect_to_weaviate_cloud(
        cluster_url=self.cluster_URL,
        auth_credentials=Auth.api_key(self.weaviate_api_key),
        skip_init_checks=True
            )
        return client 
    
    def load_vector_store_from_collection(self):    
        vector_store = WeaviateVectorStore(
        client=self.client,
        index_name=self.collection_name,
        text_key=self.text_key,
        embedding=self.embedder
        )   
        return vector_store  
    
    def get_collection(self):
        collection = self.client.collections.get(self.collection_name)
        return collection    
    
    def add_documents_data(self, html_path, metadata):
        try:
            convertHTMl = ConvertHTMLPipeline()
            json_path = convertHTMl.convert_html_file_to_json(html_file_path=html_path)
            my_documents = convertHTMl.convert_json_to_documents(json_path , metadata)
            vector_store = self.load_vector_store_from_collection()
            vector_store.add_documents(documents= my_documents)
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
        collection = self.get_collection()
        search_filter = Filter.by_property(property).like(metadata_filter)
        return collection.data.delete_many(where=search_filter)
    
    def search_documents_by_metadata(self , metadata_filter , property):
        """
        property : name (str) | active (bool) | date (str) 
        metadata_filter : the value of the property
        """
        collection = self.get_collection()
        search_filter = Filter.by_property(property).like(metadata_filter)
        result = collection.query.fetch_objects(filters=search_filter)
        chunks = []
        for o in result.objects:
                chunks.append(o.properties)
        return chunks
    
    def get_all_documents(self):
        collection = self.get_collection()
        result = collection.query.fetch_objects(limit=5000)
        for o in result.objects:
            print(o.properties)
        
        return o.properties
    
    def get_all_files_uniqe_by_name(self):
        collection = self.get_collection()
        result = collection.query.fetch_objects(limit=5000)
        unique_files = {}
        for o in result.objects:
            # Get the properties of the current object
            properties = o.properties
    
            # Exclude the 'text' key from the properties
            filtered_properties = {k: v for k, v in properties.items() if k != 'text'}
    
            # Get the name key
            name_key = filtered_properties.get('name')
    
            # Store only one entry per name
            if name_key and name_key not in unique_files:
                unique_files[name_key] = filtered_properties
        return list(unique_files.values())

