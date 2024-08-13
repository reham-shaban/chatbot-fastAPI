import os, cohere
from groq import Groq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate


load_dotenv(dotenv_path='../variables/.env')
groq_api_key = os.getenv('GROQ_API_KEY')

def message_groq(message : str):
    client = Groq(api_key=groq_api_key)
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": message,
            }
        ],
        model="gemma2-9b-it",
    )
    return chat_completion.choices[0].message.content

class RAGPipeline:
    def __init__(self, vectorstore, cohere_api_key, k, template=None):
        self.vectorstore = vectorstore
        self.k = k
        self.retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": self.k})
        self.prompt_template = PromptTemplate.from_template(template or self._get_default_template())
        self.co = cohere.Client(api_key=cohere_api_key)

    def _get_default_template(self):
        return """
        قم بفهم السياقات المقدمة مع كل سؤال ثم بقم بالاجابة على السؤال.
        أجب باللغة العربية فقط.       
        إذا كنت لا تعرف الإجابة، فقط قل إنك لا تعرف، لا تحاول تصنيع إجابة.
        في حال عدم وضوح السؤال استفسر أكثر واقترح أسئلة للتوضيح وفقاََ لفهمك والمعلومات المقدمة.
        حافظ على إجابتك شاملة وصحيحة ومختصرة قدر الإمكان.
        أضف مقدمة مناسبة تشرح للزبون ماهية سؤاله و ماهية الجواب.
        كن لبقا في إجاباتك.
        \n السياق: {context}
        \n السؤال: {question}
        \n الإجابة المفيدة:
        """

    def generate_response(self, question, conversation_id):
        try:
            retrieved_docs = self._retrieve_documents(question)
            message = self._create_prompt(retrieved_docs, question)
            response = self._query_model(message, conversation_id)
            return response
        except Exception as e:
            return f"Error generating response: {e}"

    def _retrieve_documents(self, question):
        try:
            retrieved_docs = self.retriever.invoke(question)
            return {f'doc_{i}': doc.page_content for i, doc in enumerate(retrieved_docs)}
        except Exception as e:
            raise ValueError(f"Error retrieving documents: {e}")

    def _create_prompt(self, docs, question):
        return self.prompt_template.format(context=docs, question=question)

    def _query_model(self, message, conversation_id):
        try:
            response = self.co.chat_stream(
                model="command-r-plus",
                message=message,
                preamble="أنت شات بوت تعمل كموظف خدمة زبائن لدى شركة سيرياتيل.",
                conversation_id=conversation_id,
                max_tokens=1500, # max number of generated tokens
                temperature=0.3, # Higher temperatures mean more random generations.
            )
            complete_response = ""
            for event in response:
                if event.event_type == "text-generation":
                    print(event.text, end="")
                    complete_response += event.text
                elif event.event_type == "stream-end":
                    print()
            return complete_response
        except Exception as e:
            raise ValueError(f"Error querying model: {e}")
