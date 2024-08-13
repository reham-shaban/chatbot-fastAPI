import os
from groq import Groq
from dotenv import load_dotenv

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