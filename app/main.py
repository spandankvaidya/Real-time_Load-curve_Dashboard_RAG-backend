# app/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Import the simplified functions from our logic files
from dash_app import create_dash_app
from chatbot_logic import generate_answer # <-- Import the new function

class ChatQuery(BaseModel):
    question: str

# The lifespan manager and app_state are no longer needed.
# We can declare the FastAPI app directly.
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For production, restrict to your Vercel URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return HTMLResponse(content="<h3>✅ FastAPI backend is running.</h3>")

@app.post("/ask-chatbot")
async def ask_chatbot(query: ChatQuery):
    # Directly call the new, simplified generate_answer function.
    answer = await generate_answer(query.question)
    return {"answer": answer}

# Mount the Dash app (this part does not need to change)
dash_app = create_dash_app()
app.mount("/dashboard", WSGIMiddleware(dash_app.server))
