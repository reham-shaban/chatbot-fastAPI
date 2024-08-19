from langchain_core.prompts import PromptTemplate
from typing import AsyncGenerator
import cohere, os

def load_template_from_file():
    try:
        # Define the path to the configuration file
        config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../config/prompt_template.txt'))
        # Open the file with the correct encoding
        with open(config_path, 'r', encoding='utf-8') as file:
            return file.read()
    except UnicodeDecodeError:
        # Handle cases where the encoding might not be UTF-8
        raise ValueError("Unable to decode the template file. Please check the file encoding.")
    except Exception as e:
        # Handle other potential exceptions
        raise ValueError(f"Error reading template file: {e}")


class RAGPipeline:
    def __init__(self, vectorstore,  conversation_id, cohere_api_key, template=None, k=20):
        self.vectorstore = vectorstore 
        self.k = k
        self.retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": self.k})
        self.prompt_template = PromptTemplate.from_template(template or self._get_default_template())
        self.co = cohere.Client(api_key=cohere_api_key)
        self.conversation_id = conversation_id

    def _get_default_template(self):
        return load_template_from_file()

    def generate_response(self, question):
        try:
            retrieved_docs = self._retrieve_documents(question)
            message = self._create_prompt(retrieved_docs, question)
            response = self._query_model(message)
            return response
        except Exception as e:
            return f"Error generating response: {e}"

    def _retrieve_documents(self, question):
        try:
            retrieved_docs = self.retriever.invoke(question)
            #print(retrieved_docs)
            return {f'doc_{i}': doc.page_content for i, doc in enumerate(retrieved_docs)}
        except Exception as e:
            raise ValueError(f"Error retrieving documents: {e}")

    def _create_prompt(self, docs, question):
        return self.prompt_template.format(context=docs, question=question)

    async def _query_model(self, message: str) -> AsyncGenerator[str, None]:
        try:
            response = self.co.chat_stream(
                model="command-r-plus",
                message=message,
                preamble="أنت شات بوت تعمل كموظف خدمة زبائن لدى شركة سيرياتيل.",
                conversation_id=self.conversation_id,
                max_tokens=1500,  # max number of generated tokens
                temperature=0.7,  # Higher temperatures mean more random generations.
            )
            async for event in response:
                if event.event_type == "text-generation":
                    yield event.text
                elif event.event_type == "stream-end":
                    break
        except Exception as e:
            yield f"Error querying model: {e}"