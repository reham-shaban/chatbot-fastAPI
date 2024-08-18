from langchain_core.prompts import PromptTemplate
import cohere

class RAGPipeline:
    def __init__(self, vectorstore,  conversation_id, cohere_api_key, template=None, k=20):
        self.vectorstore = vectorstore 
        self.k = k
        self.retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": self.k})
        self.prompt_template = PromptTemplate.from_template(template or self._get_default_template())
        self.co = cohere.Client(api_key=cohere_api_key)
        self.conversation_id = conversation_id

    def _get_default_template(self):
        return """
        قم بفهم السياقات المقدمة مع كل سؤال ثم بقم بالاجابة على السؤال.
        أجب باللغة العربية فقط.       
        
        إذا كنت لا تعرف الإجابة، فقط قل إنك لا تعرف، لا تحاول تصنيع إجابة.
        سيصلك العديد من Documents لا توجد صلة وصل بينهم إلا إذا كانت الmetadata تحوي نفس ال name
        اذا كانت الإجابة من جدول ما أرسله بالكامل
        في حال عدم وضوح السؤال استفسر أكثر واقترح أسئلة للتوضيح وفقاََ لفهمك والمعلومات المقدمة.
        بعد الإجابة قم باقتراح ثلاث أسئلة من المعلومات المقدمة لك
        حافظ على إجابتك شاملة وصحيحة ومختصرة قدر الإمكان.
        أضف مقدمة مناسبة تشرح للزبون ماهية سؤاله و ماهية الجواب.
        كن لبقا في إجاباتك.
        \n السياق: {context}
        \n السؤال: {question}
        \n الإجابة المفيدة:
        """

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

    def _query_model(self, message):
        try:
            response = self.co.chat_stream(
                model="command-r-plus",
                message=message,
                preamble="أنت شات بوت تعمل كموظف خدمة زبائن لدى شركة سيرياتيل.",
                conversation_id=self.conversation_id,
                max_tokens=1500, # max number of generated tokens
                temperature=0.7, # Higher temperatures mean more random generations.
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
