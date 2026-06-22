import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from researcher import run_research

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
def research(request: ResearchRequest):
    return {"report": run_research(request.topic, request.session_id)}

@app.get("/health")
def health_check():
    return{"status": "ok"}