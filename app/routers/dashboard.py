from fastapi import APIRouter, HTTPException
from app.services.documents import convert_html_file_to_json
# from app.services.vectorstore import (
#     load_json_file,
#     clean_dict_strings
# )

router = APIRouter()

@router.post("/convert")
async def convert_html(html_file_path: str, json_file_path: str):
    try:
        convert_html_file_to_json(html_file_path, json_file_path)
        return {"message": "HTML file converted to JSON successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))