from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
import time
import jwt  # pip install PyJWT

router = APIRouter()

OPENAI_MODEL = "gpt-4o-realtime-preview"


class LogEntry(BaseModel):
    role: str
    text: str
    raw: dict


@router.get("/openai/voice-session-token")
def get_voice_session_token():
    """Returns a JWT that allows the browser to open a ws:// session
    without exposing the API key.
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key or not isinstance(openai_api_key, str):
        raise HTTPException(
            status_code=500,
            detail="Server missing OPENAI_API_KEY for realtime voice token. Set it in the environment/.env.",
        )

    payload = {
        "iss": "fastapi-backend",
        "aud": "openai-realtime-api",
        "model": OPENAI_MODEL,
        "exp": int(time.time()) + 60,  # 1 min expiry
    }

    token = jwt.encode(payload, openai_api_key, algorithm="HS256")
    return {"token": token}


@router.post("/log-voice-interaction")
def log_voice_interaction(entry: LogEntry):
    """Save text + metadata to DB."""
    # TODO: replace with your DB ORM
    print("LOG >>", entry.role, entry.text)

    # e.g. db.add_conversation_line(...)
    return {"ok": True}
