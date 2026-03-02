from fastapi import APIRouter

router = APIRouter(prefix="/languages", tags=["Languages"])

SUPPORTED_LANGUAGES = [
    {"code": "eng", "name": "English"},
    {"code": "hin", "name": "Hindi/Devanagari"},
    {"code": "mar", "name": "Marathi"}
]

@router.get("/")
async def get_languages():
    return SUPPORTED_LANGUAGES
