import requests
import json
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# Get API key
api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    print("Error: OPENROUTER_API_KEY not found in .env file")
    print("Please make sure your .env file contains: OPENROUTER_API_KEY=your_key_here")
    exit(1)

print("API Key found. Making request to OpenRouter...")

try:
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "Career Survey Chatbot",
        },
        json={
            "model": "google/gemma-3n-e4b-it:free",
            "messages": [
                {
                    "role": "user",
                    "content": "What is the meaning of life?"
                }
            ]
        }
    )
    
    print("\n--- Response Status Code:", response.status_code)
    print("--- Response Headers:", response.headers)
    print("\n--- Response Body:")
    print(json.dumps(response.json(), indent=2))
    
except requests.exceptions.RequestException as e:
    print(f"\n--- Request Failed ---")
    print(f"Error: {str(e)}")
    
    if hasattr(e, 'response') and e.response is not None:
        print(f"Status Code: {e.response.status_code}")
        print("Response:", e.response.text)