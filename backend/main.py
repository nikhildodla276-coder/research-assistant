import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from researcher import run_research

load_dotenv()

app = FastAPI()

class ResearchRequest(BaseModel):
    topic: str
    session_id:str="default"

@app.post("/research")
def research(request: ResearchRequest):
    return {"report": run_research(request.topic, request.session_id)}

@app.get("/health")
def health_check():
    return{"status": "ok"}