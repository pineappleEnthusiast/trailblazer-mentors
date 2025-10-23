# Career Survey Chatbot

An LLM-powered chatbot that helps professionals reflect on their career journey through an interactive conversation.

## Features

- Interactive chat interface
- Stores conversation history in SQLite database
- Responsive design that works on desktop and mobile
- Easy to deploy and customize

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables**:
   Create a `.env` file in the root directory with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

3. **Run the application**:
   ```bash
   uvicorn app.main:app --reload
   ```

4. **Access the application**:
   Open your browser and go to `http://localhost:8000`

## Project Structure

- `app/main.py` - Main FastAPI application
- `app/templates/` - HTML templates
- `app/static/` - Static files (CSS, JS, images)
- `career_survey.db` - SQLite database (created automatically)

## Next Steps

1. **Integrate with an LLM API** (currently uses a simple response system)
2. Add user authentication
3. Add more sophisticated conversation analysis
4. Export conversation data

## License

MIT
