from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.documents import html_to_json

router = APIRouter()

@router.post("/html-to-json")
async def convert_html(file: UploadFile = File(...)):
    try:
        # Read the HTML content from the uploaded file
        html_content = await file.read()

        # Convert HTML to JSON content
        json_content = html_to_json(html_content.decode("utf-8"))

        # Return the JSON content as a response
        return json_content

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))