from fastapi import FastAPI, Request, HTTPException, Depends, UploadFile, File, Form, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import sqlite3
import json
import os
import asyncio
import websockets
import requests
from openai import OpenAI
from app.routes.openai import router as openai_router
from dotenv import load_dotenv
from datetime import datetime
import os
import tempfile
import uuid

# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
print(f"Loading .env from: {os.path.abspath(env_path)}")
load_dotenv(dotenv_path=env_path)

# Configure OpenAI (do not crash if missing; handle at request time)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")

openai_client = None

if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY not found in environment variables. OpenAI will be unavailable.")
else:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Debug info
print("\nLLM Configuration:")
print(f"OpenAI key present: {bool(OPENAI_API_KEY)}")
print(f"OpenAI model override: {OPENAI_MODEL if OPENAI_MODEL else 'None'}")

async def get_llm_response(messages: List[Dict[str, str]]) -> str:
    """Get response from the configured LLM (OpenAI preferred, Groq as fallback)"""
    try:
        print("\n--- Preparing LLM request ---")
        print(f"Messages being sent: {json.dumps(messages, indent=2)}")

        # Prefer OpenAI if configured
        if openai_client is not None:
            preferred_models = [
                "gpt-5-nano",
                "gpt-4o-mini",
                "gpt-4o"
            ]

            # Allow override via env var, else select first default
            model_to_use = OPENAI_MODEL if OPENAI_MODEL else preferred_models[0]
            print(f"\n--- Using OpenAI model: {model_to_use} ---")

            # Convert ChatML messages to Responses API input format
            # Rules:
            # - system -> developer with input_text
            # - user -> user with input_text
            # - assistant -> assistant with output_text
            inputs = []
            for m in messages:
                role = (m.get("role") or "user").strip()
                text = (m.get("content") or "").strip()
                if not text:
                    continue
                if role == "system":
                    role = "developer"
                content_type = "output_text" if role == "assistant" else "input_text"
                inputs.append({
                    "role": role,
                    "content": [
                        {"type": content_type, "text": text}
                    ]
                })

            response = openai_client.responses.create(
                model=model_to_use,
                input=inputs,
                text={"format": {"type": "text"}}
            )

            # Prefer output_text convenience field
            response_content = getattr(response, "output_text", None)

            # Fallback: traverse structured output to gather text
            if not response_content:
                parts = []
                try:
                    for item in getattr(response, "output", []) or []:
                        for c in getattr(item, "content", []) or []:
                            ctype = getattr(c, "type", None)
                            if ctype in ("output_text", "input_text", "text"):
                                text_obj = getattr(c, "text", None)
                                value = getattr(text_obj, "value", None) if text_obj else None
                                if value:
                                    parts.append(value)
                except Exception:
                    pass
                response_content = "\n".join(parts).strip() if parts else None

            if not response_content:
                raise ValueError("Empty response from OpenAI")
        else:
            raise ValueError("No LLM configured. Please set OPENAI_API_KEY.")

        # Clean up the response: trim whitespace only
        response_content = response_content.strip()

        print(f"\n--- Received Response ---")
        print(f"Response: {response_content}")
        return response_content
        
    except Exception as e:
        error_msg = f"Error with LLM API: {str(e)}"
        print(f"\n--- Error ---\n{error_msg}")
        import traceback
        traceback.print_exc()
        # Surface the error message to assist debugging
        return f"I'm having trouble generating a response. ({error_msg})"

app = FastAPI(title="Career Survey Chatbot")
app.include_router(openai_router)

# Configure CORS
origins = [
    "http://localhost:5173",  # React dev server
    "http://localhost:8000",  # Local FastAPI template host
    "http://127.0.0.1:8000",
    "https://framer.com",     # Framer preview
    "https://trailblazernetwork.framer.ai",  # Framer production
]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (use absolute paths; guard if missing in prod)
BASE_DIR = os.path.dirname(__file__)
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
else:
    print(f"Warning: Static directory not found at {STATIC_DIR}, skipping mount.")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Database setup
DATABASE_URL = "sqlite:///./career_survey.db"

def get_db():
    conn = sqlite3.connect('career_survey.db')
    return conn

def init_db():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        conversation_data TEXT
    )
    ''')
    # Store uploaded audio clips and their transcripts
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS audio_clips (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        participant_id INTEGER,
        filename TEXT,
        content_type TEXT,
        audio BLOB,
        transcript TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    # Create participants table to store user info
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS participants (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        first_name TEXT,
        last_name TEXT,
        email TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    # Voice sessions table for mentor realtime interviews
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS voice_sessions (
        session_id TEXT PRIMARY KEY,
        first_name TEXT,
        last_name TEXT,
        email TEXT,
        transcript TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    # Ensure conversations table has participant_id column
    cursor.execute("PRAGMA table_info(conversations)")
    columns = [row[1] for row in cursor.fetchall()]
    if 'participant_id' not in columns:
        cursor.execute('ALTER TABLE conversations ADD COLUMN participant_id INTEGER')
    conn.commit()
    conn.close()

# Initialize database on startup
init_db()

class Message(BaseModel):
    role: str
    content: str

class Conversation(BaseModel):
    messages: List[Message] = []
    participant_id: Optional[int] = None

class Registration(BaseModel):
    first_name: str
    last_name: str
    email: str


class MentorInfo(BaseModel):
    first_name: str
    last_name: str
    email: str


class SaveTranscriptRequest(BaseModel):
    text: str

# Initial system message that sets up the chatbot's behavior
SYSTEM_PROMPT = """You are a friendly and professional career survey chatbot. Your goal is to help professionals reflect on their career journey by asking thoughtful questions.

Guidelines:
1. Start by introducing yourself and explaining the purpose of the survey
2. Ask one question at a time
3. Show genuine interest in their responses
4. Ask follow-up questions based on their answers
5. Be supportive and encouraging
6. Keep the conversation natural and conversational
7. When appropriate, ask about their job role, daily tasks, skills used, challenges, and what they enjoy
8. End the conversation when the user indicates they're done

Example questions:
- Can you tell me about your current role and what a typical day looks like for you?
- What skills do you find most valuable in your position?
- What aspects of your job do you find most rewarding?
- What kind of person do you think would thrive in this role?
"""

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/voice", response_class=HTMLResponse)
async def voice_view(request: Request):
    """Standalone view that hosts the React-based Voice Interview Agent."""
    return templates.TemplateResponse("voice.html", {"request": request})


@app.websocket("/voice-ws")
async def voice_ws(browser_ws: WebSocket):
    await browser_ws.accept()

    if not OPENAI_API_KEY:
        await browser_ws.close(code=1011)
        return

    openai_url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview"
    openai_ws = await websockets.connect(
        openai_url,
        additional_headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        max_size=2 ** 26,
    )

    async def forward_browser_to_openai():
        try:
            while True:
                msg = await browser_ws.receive()
                if "bytes" in msg and msg["bytes"] is not None:
                    await openai_ws.send(msg["bytes"])
                elif "text" in msg and msg["text"] is not None:
                    await openai_ws.send(msg["text"])
        except WebSocketDisconnect:
            await openai_ws.close()
        except Exception:
            await openai_ws.close()

    async def forward_openai_to_browser():
        try:
            async for message in openai_ws:
                if isinstance(message, bytes):
                    await browser_ws.send_bytes(message)
                else:
                    await browser_ws.send_text(message)
        except Exception:
            try:
                await browser_ws.close()
            except Exception:
                pass

    await asyncio.gather(
        forward_browser_to_openai(),
        forward_openai_to_browser(),
    )


@app.get("/mentor/webrtc-token")
async def get_webrtc_token():
    """Generate an ephemeral Realtime session for WebRTC clients."""
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="Server missing OPENAI_API_KEY for WebRTC token.")

    try:
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        resp = requests.post(
            "https://api.openai.com/v1/realtime/sessions",
            headers=headers,
            json={
                "model": "gpt-4o-realtime-preview",
                "voice": "verse",
                "modalities": ["audio", "text"],
                "instructions": (
                    "You are a friendly and professional interviewer who only speaks English. "
                    "Your sole job is to interview people about their career journeys. "
                    "Ask one thoughtful question at a time, listen carefully, and ask relevant follow-up "
                    "questions about their roles, skills, challenges, and what they enjoy. "
                    "Do not switch languages or topics away from their career journey."
                ),
            },
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Failed to create Realtime session: {str(e)}")


@app.post("/mentor/start-session")
async def mentor_start_session(info: MentorInfo):
    """Create a new mentor voice session and return its session_id."""
    session_id = str(uuid.uuid4())
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO voice_sessions (session_id, first_name, last_name, email, transcript) VALUES (?, ?, ?, ?, ?)',
            (session_id, info.first_name.strip(), info.last_name.strip(), info.email.strip(), "")
        )
        conn.commit()
        return {"session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start session: {str(e)}")
    finally:
        if 'conn' in locals():
            conn.close()


@app.post("/mentor/save-transcript/{session_id}")
async def mentor_save_transcript(session_id: str, payload: SaveTranscriptRequest):
    """Persist the transcript for a given mentor voice session."""
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute(
            'UPDATE voice_sessions SET transcript = ? WHERE session_id = ?',
            (payload.text, session_id)
        )
        conn.commit()
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save transcript: {str(e)}")
    finally:
        if 'conn' in locals():
            conn.close()


@app.post("/api/register")
async def register_user(reg: Registration):
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO participants (first_name, last_name, email) VALUES (?, ?, ?)',
            (reg.first_name.strip(), reg.last_name.strip(), reg.email.strip())
        )
        participant_id = cursor.lastrowid
        conn.commit()
        return {"participant_id": participant_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to register: {str(e)}")
    finally:
        if 'conn' in locals():
            conn.close()

@app.post("/api/chat")
async def chat(conversation: Conversation, background_tasks: BackgroundTasks):
    print("\n--- New Chat Request ---")
    print(f"Received messages: {[f'{m.role}: {m.content[:50]}...' for m in conversation.messages]}")
    
    try:
        # Check for exit commands
        last_message = conversation.messages[-1].content.lower()
        if any(word in last_message for word in ["bye", "goodbye", "exit", "quit"]):
            print("Ending conversation as requested by user")
            save_conversation(conversation)
            return {"response": "Thank you for sharing your career journey with me! Your insights are valuable and have been recorded. Have a great day!"}
        
        # Prepare the conversation context
        system_content = """You are a friendly and professional career survey chatbot. Your goal is to help professionals reflect on their career journey by asking thoughtful questions.

Guidelines:
1. Keep responses concise and focused on one question or follow-up at a time
2. Show genuine interest in their responses
3. Ask relevant follow-up questions based on their previous answers
4. Be supportive and encouraging
5. Keep the conversation natural and conversational
6. When appropriate, ask about their job role, daily tasks, skills used, challenges, and what they enjoy
7. End the conversation when the user indicates they're done"""
            
        # Start with system message and conversation history
        messages = [
            {"role": "system", "content": system_content}
        ]
        
        # Add conversation history (most recent 4 messages to reduce latency)
        for msg in conversation.messages[-4:]:
            if msg.role != 'system':  # Skip any system messages that might be in the history
                messages.append({"role": msg.role, "content": msg.content})
        
        print(f"Sending to LLM: {json.dumps(messages, indent=2)}")
        
        # Get response from LLM
        if openai_client is None:
            raise ValueError("Server missing LLM configuration. Set OPENAI_API_KEY.")
        response = await get_llm_response(messages)
        
        # Ensure we have a valid response
        if not response or not response.strip():
            raise ValueError("Received empty response from LLM")
            
        print(f"LLM Response: {response}")
        # Persist the conversation including the assistant's reply in background
        full_messages = [m.dict() for m in conversation.messages]
        full_messages.append({"role": "assistant", "content": response})
        tmp_conv = Conversation(messages=[Message(**m) for m in full_messages], participant_id=getattr(conversation, 'participant_id', None))
        background_tasks.add_task(save_conversation, tmp_conv)
        return {"response": response}
        
    except Exception as e:
        error_msg = f"Error in chat endpoint: {str(e)}"
        print(error_msg)
        return {
            "response": "I'm having some technical difficulties. Let me rephrase my last question: " + 
                       "Could you tell me more about your current role and what you enjoy most about it?"
        }

def save_conversation(conversation: Conversation):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO conversations (conversation_data, participant_id) VALUES (?, ?)',
        (
            json.dumps([msg.dict() for msg in conversation.messages]),
            conversation.participant_id if hasattr(conversation, 'participant_id') else None
        )
    )
    conn.commit()
    conn.close()

@app.get("/api/conversations")
async def get_conversations():
    """
    Retrieve all conversations from the database.
    Returns a list of conversations with their IDs and timestamps.
    """
    try:
        conn = get_db()
        cursor = conn.cursor()
        
        # Fetch all conversations with their metadata
        cursor.execute('''
            SELECT id, created_at, conversation_data 
            FROM conversations 
            ORDER BY created_at DESC
        ''')
        
        conversations = []
        for row in cursor.fetchall():
            conv_id, created_at, conv_data = row
            conversations.append({
                'id': conv_id,
                'created_at': created_at,
                'message_count': len(json.loads(conv_data)) if conv_data else 0,
                'preview': json.loads(conv_data)[0]['content'][:100] + '...' if conv_data else ''
            })
            
        return {"conversations": conversations}
    
    except Exception as e:
        print(f"Error fetching conversations: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching conversations")
    
    finally:
        if 'conn' in locals():
            conn.close()

@app.post("/api/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    participant_id: Optional[int] = Form(None),
    session_id: Optional[str] = Form(None),
):
    """Transcribe uploaded audio using OpenAI gpt-4o-transcribe model."""
    try:
        if openai_client is None:
            raise HTTPException(status_code=400, detail="OPENAI_API_KEY not configured on server")

        # Persist to a temporary file because OpenAI SDK expects a file-like object
        suffix = os.path.splitext(file.filename or "audio.webm")[1] or ".webm"
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp_path = tmp.name
                contents = await file.read()
                tmp.write(contents)
                tmp.flush()

            # Reopen after the NamedTemporaryFile context (Windows compatibility)
            with open(tmp_path, "rb") as f:
                try:
                    # Preferred modern model
                    transcription = openai_client.audio.transcriptions.create(
                        model="gpt-4o-transcribe",
                        file=f,
                    )
                except Exception as primary_err:
                    print(f"Primary transcription failed (gpt-4o-transcribe): {primary_err}")
                    # Fallback to whisper-1 for broader availability
                    f.seek(0)
                    transcription = openai_client.audio.transcriptions.create(
                        model="whisper-1",
                        file=f,
                    )
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception as cleanup_err:
                    print(f"Warning: failed to remove temp file {tmp_path}: {cleanup_err}")

        text = getattr(transcription, "text", None)
        if not text and isinstance(transcription, dict):
            text = transcription.get("text")

        # Persist audio and transcript regardless of transcription result
        try:
            conn = get_db()
            cursor = conn.cursor()

            # Store raw audio clip
            cursor.execute(
                'INSERT INTO audio_clips (participant_id, filename, content_type, audio, transcript) VALUES (?, ?, ?, ?, ?)',
                (
                    participant_id,
                    file.filename,
                    file.content_type,
                    sqlite3.Binary(contents) if 'contents' in locals() else None,
                    text
                )
            )

            # If this transcription is associated with a mentor voice session, update its transcript
            if text and session_id:
                cursor.execute(
                    'UPDATE voice_sessions SET transcript = ? WHERE session_id = ?',
                    (text, session_id)
                )

            conn.commit()
        finally:
            if 'conn' in locals():
                conn.close()

        if not text:
            raise HTTPException(status_code=500, detail="No transcription text returned")

        return {"text": text}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Transcription error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to transcribe audio: {str(e)}")

@app.get("/db", response_class=HTMLResponse)
async def view_database():
    """Simple HTML view of key database tables with audio download links."""
    try:
        conn = get_db()
        cursor = conn.cursor()

        # Fetch participants
        cursor.execute("SELECT id, first_name, last_name, email, created_at FROM participants ORDER BY created_at DESC")
        participants = cursor.fetchall()

        # Fetch conversations (include data for transcript render)
        cursor.execute("SELECT id, created_at, conversation_data, participant_id FROM conversations ORDER BY created_at DESC")
        conv_rows = cursor.fetchall()
        conversations = []
        for r in conv_rows:
            conv_id, created_at, conv_data, pid = r
            msgs = []
            try:
                msgs = json.loads(conv_data) if conv_data else []
            except Exception:
                msgs = []
            conversations.append({
                'id': conv_id,
                'created_at': created_at,
                'participant_id': pid,
                'message_count': len(msgs),
                'messages': msgs,
            })

        # Fetch audio clips
        cursor.execute("SELECT id, participant_id, filename, content_type, LENGTH(audio), created_at FROM audio_clips ORDER BY created_at DESC")
        audio_clips = cursor.fetchall()

        # Fetch voice sessions
        cursor.execute("SELECT session_id, first_name, last_name, email, transcript, created_at FROM voice_sessions ORDER BY created_at DESC")
        voice_sessions = cursor.fetchall()

        # Build HTML
        def esc(x):
            return (str(x) if x is not None else '').replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')

        html = [
            "<html><head><title>DB Viewer</title>",
            "<style>body{font-family:sans-serif;padding:16px} table{border-collapse:collapse;margin:16px 0;width:100%} th,td{border:1px solid #ddd;padding:8px;font-size:14px} th{background:#f3f4f6;text-align:left} h2{margin-top:24px}</style>",
            "</head><body>",
            "<h1>Database Viewer</h1>",
        ]

        # Participants
        html.append("<h2>Participants</h2>")
        html.append("<table><tr><th>ID</th><th>Name</th><th>Email</th><th>Created</th></tr>")
        for p in participants:
            pid, fn, ln, em, created = p
            html.append(f"<tr><td>{pid}</td><td>{esc(fn)} {esc(ln)}</td><td>{esc(em)}</td><td>{esc(created)}</td></tr>")
        html.append("</table>")

        # Conversations
        html.append("<h2>Conversations</h2>")
        html.append("<table><tr><th>ID</th><th>Participant</th><th>Messages</th><th>Created</th><th>Transcript</th></tr>")
        for c in conversations:
            # Build transcript HTML
            transcript_parts = ["<details><summary>View transcript</summary><div style='padding:8px 0'>"]
            if c['messages']:
                transcript_parts.append("<ol style='margin:0;padding-left:20px'>")
                for m in c['messages']:
                    role = esc(m.get('role'))
                    content = esc(m.get('content'))
                    transcript_parts.append(f"<li><strong>{role}:</strong> {content}</li>")
                transcript_parts.append("</ol>")
            else:
                transcript_parts.append("<em>No messages</em>")
            transcript_parts.append("</div></details>")
            transcript_html = "".join(transcript_parts)

            html.append(
                f"<tr><td>{c['id']}</td><td>{esc(c['participant_id'])}</td><td>{c['message_count']}</td><td>{esc(c['created_at'])}</td><td>{transcript_html}</td></tr>"
            )
        html.append("</table>")

        # Audio clips
        html.append("<h2>Audio Clips</h2>")
        html.append("<table><tr><th>ID</th><th>Participant</th><th>Filename</th><th>Type</th><th>Size (bytes)</th><th>Created</th><th>Download</th></tr>")
        for a in audio_clips:
            aid, pid, fname, ctype, size, created = a
            is_mp3 = (ctype == 'audio/mpeg') or (str(fname or '').lower().endswith('.mp3'))
            label = 'Download MP3' if is_mp3 else 'Download'
            html.append(
                f"<tr><td>{aid}</td><td>{esc(pid)}</td><td>{esc(fname)}</td><td>{esc(ctype)}</td><td>{size or 0}</td><td>{esc(created)}</td>"
                f"<td><a href=\"/api/audio/{aid}\">{label}</a></td></tr>"
            )
        html.append("</table>")

        # Voice sessions
        html.append("<h2>Voice Sessions</h2>")
        html.append("<table><tr><th>Session ID</th><th>Name</th><th>Email</th><th>Created</th><th>Transcript</th></tr>")
        for vs in voice_sessions:
            sid, fn, ln, em, transcript, created = vs
            transcript_html = esc(transcript) if transcript else "<em>No transcript</em>"
            html.append(
                f"<tr><td>{esc(sid)}</td><td>{esc(fn)} {esc(ln)}</td><td>{esc(em)}</td><td>{esc(created)}</td><td>{transcript_html}</td></tr>"
            )
        html.append("</table>")

        html.append("</body></html>")
        return HTMLResponse("".join(html))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if 'conn' in locals():
            conn.close()

@app.get("/api/audio/{clip_id}")
async def download_audio(clip_id: int):
    """Download the stored audio blob by id with appropriate content type and filename."""
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT filename, content_type, audio FROM audio_clips WHERE id = ?", (clip_id,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Audio clip not found")
        filename, content_type, audio_blob = row
        if audio_blob is None:
            raise HTTPException(status_code=404, detail="No audio data available")
        # Sensible defaults
        content_type = content_type or 'application/octet-stream'
        download_name = filename or f"clip_{clip_id}.bin"
        headers = {
            "Content-Disposition": f"attachment; filename=\"{download_name}\""
        }
        return Response(content=audio_blob, media_type=content_type, headers=headers)
    finally:
        if 'conn' in locals():
            conn.close()

@app.get("/api/debug/db")
async def debug_database():
    """Debug endpoint to view the complete database contents"""
    try:
        conn = get_db()
        cursor = conn.cursor()
        
        # Get all tables in the database
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        result = {}
        for table in tables:
            table_name = table[0]
            # Get all rows from the table
            cursor.execute(f"SELECT * FROM {table_name}")
            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()
            
            # Convert rows to list of dicts
            table_data = []
            for row in rows:
                row_data = {}
                for i, value in enumerate(row):
                    # Try to parse JSON data if the column name suggests it might be JSON
                    if columns[i] == 'conversation_data' and value:
                        try:
                            row_data[columns[i]] = json.loads(value)
                        except:
                            row_data[columns[i]] = value
                    else:
                        row_data[columns[i]] = value
                table_data.append(row_data)
                
            result[table_name] = table_data
            
        return result
        
    except Exception as e:
        return {"error": str(e)}
        
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
