from pydantic import BaseModel

class ChatMessage(BaseModel):
    message: str

class Metadata(BaseModel):
    name: str
    active: bool
    date: str