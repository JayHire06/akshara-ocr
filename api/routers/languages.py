from fastapi import APIRouter

router = APIRouter(prefix="/languages", tags=["Languages"])

SUPPORTED_LANGUAGES = [
    {"code": "hin", "name": "Hindi/Devanagari"},
    {"code": "ben", "name": "Bengali"},
    {"code": "tam", "name": "Tamil"},
    {"code": "tel", "name": "Telugu"},
    {"code": "kan", "name": "Kannada"},
    {"code": "mal", "name": "Malayalam"},
    {"code": "mar", "name": "Marathi"},
    {"code": "guj", "name": "Gujarati"},
    {"code": "pun", "name": "Punjabi"},
    {"code": "ori", "name": "Odia"},
    {"code": "eng", "name": "English"}
]

@router.get("/")
async def get_languages():
    return SUPPORTED_LANGUAGES
