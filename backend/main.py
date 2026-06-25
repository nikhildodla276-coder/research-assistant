import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from discord_notifier import notify_discord
from researcher import run_research, run_chat

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ResearchRequest(BaseModel):
    topic: str
    session_id:str="default"

@app.post("/research")
async def research(request: ResearchRequest):
    result = await run_research(request.topic, request.session_id)
    await notify_discord(request.topic, request.session_id)
    return {"report": result}

@app.get("/health")
def health_check():
    return{"status": "ok"}

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


@app.post("/chat")
async def chat(request: ChatRequest):
    response = await run_chat(request.message, request.session_id)
    return {"response": response}