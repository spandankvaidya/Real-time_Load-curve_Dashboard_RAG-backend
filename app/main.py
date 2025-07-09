from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# --- NEW IMPORTS for the chatbot ---
from app.chatbot_logic import get_chatbot_response

# --- EXISTING IMPORTS for the dash app ---
from app.dash_app import create_dash_app

app = FastAPI()

# CORS setup remains the same. This allows your Vercel frontend to call the backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For production, restrict this to your Vercel URL e.g., ["https://your-frontend.vercel.app"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#               NEW CHATBOT ENDPOINT

# Pydantic model to define and validate the structure of the incoming request body.
class ChatRequest(BaseModel):
    question: str

@app.post("/ask-chatbot")
async def ask_chatbot(request: ChatRequest):
    """
    This endpoint receives a question from the frontend, gets a response
    from the Groq LLM via our chatbot logic, and returns the answer.
    """
    try:
        user_question = request.question
        # Call the function from chatbot.py that invokes the LLM
        answer = get_chatbot_response(user_question)
        # Return the answer in the JSON format the frontend expects
        return {"answer": answer}
    except Exception as e:
        print(f"Error in /ask-chatbot endpoint: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing the chatbot request.")

#               EXISTING DASHBOARD LOGIC (Unchanged)

@app.get("/")
def root():
    return HTMLResponse(content="<h3>✅ FastAPI backend with mounted Dash app and Chatbot is running.</h3>")

# Mount the Dash app under the /dashboard prefix. This logic remains the same.
dash_app = create_dash_app()
app.mount("/dashboard", WSGIMiddleware(dash_app.server))
