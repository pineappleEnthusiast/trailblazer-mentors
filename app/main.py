from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import sqlite3
import json
import os
from groq import Groq
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
print(f"Loading .env from: {os.path.abspath(env_path)}")
load_dotenv(dotenv_path=env_path)

# Configure Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables")

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# Debug info
print("\nGroq Configuration:")
print(f"API Key present: {bool(GROQ_API_KEY)}")

async def get_llm_response(messages: List[Dict[str, str]]) -> str:
    """Get response from Groq's LLM"""
    try:
        print("\n--- Sending to Groq ---")
        print(f"Messages being sent: {json.dumps(messages, indent=2)}")
        
        # List available models
        available_models = client.models.list()
        model_names = [model.id for model in available_models.data]
        print(f"\n--- Available Models ---")
        print("\n".join(model_names))
        
        # Try to find a suitable model
        preferred_models = [
            "mixtral-8x7b-32768",
            "llama3-70b-8192",
            "llama3-8b-8192"
        ]
        
        model_to_use = None
        for model in preferred_models:
            if model in model_names:
                model_to_use = model
                break
                
        if not model_to_use and model_names:
            model_to_use = model_names[0]  # Fallback to first available model
            
        if not model_to_use:
            raise ValueError("No available models found")
            
        print(f"\n--- Using model: {model_to_use} ---")
        
        # Groq API call
        response = client.chat.completions.create(
            model=model_to_use,
            messages=messages,
            temperature=0.7,
            max_tokens=512,
        )
        
        response_content = response.choices[0].message.content
        if not response_content:
            raise ValueError("Empty response from Groq")
            
        # Clean up the response to remove any internal thinking
        if '\n\n' in response_content:
            # If there's a double newline, take everything after the last one
            response_content = response_content.split('\n\n')[-1]
            
        # Remove any leading/trailing whitespace and quotes
        response_content = response_content.strip().strip('"\'')
            
        print(f"\n--- Received Response ---")
        print(f"Response: {response_content}")
        return response_content
        
    except Exception as e:
        error_msg = f"Error with Groq API: {str(e)}"
        print(f"\n--- Error ---\n{error_msg}")
        import traceback
        traceback.print_exc()
        return "I'm having trouble generating a response. Please try again in a moment."

app = FastAPI(title="Career Survey Chatbot")

# Configure CORS
origins = [
    "http://localhost:5173",  # React dev server
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
    conn.commit()
    conn.close()

# Initialize database on startup
init_db()

class Message(BaseModel):
    role: str
    content: str

class Conversation(BaseModel):
    messages: List[Message] = []

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

@app.post("/api/chat")
async def chat(conversation: Conversation):
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
        
        # Add conversation history (most recent 6 messages)
        for msg in conversation.messages[-6:]:
            if msg.role != 'system':  # Skip any system messages that might be in the history
                messages.append({"role": msg.role, "content": msg.content})
        
        print(f"Sending to LLM: {json.dumps(messages, indent=2)}")
        
        # Get response from LLM
        response = await get_llm_response(messages)
        
        # Ensure we have a valid response
        if not response or not response.strip():
            raise ValueError("Received empty response from LLM")
            
        print(f"LLM Response: {response}")
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
        'INSERT INTO conversations (conversation_data) VALUES (?)',
        (json.dumps([msg.dict() for msg in conversation.messages]),)
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
