from pydantic import BaseModel

class ChatMessage(BaseModel):
    message: str

class MetadataFilter(BaseModel):
    property: str
    metadata_filter: str