import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from researcher import run_research

load_dotenv()

app = FastAPI()

class ResearchRequest(BaseModel):
    topic: str

@app.post("/research")
def research(request: ResearchRequest):
    return {"report": run_research(request.topic)}

@app.get("/health")
def health_check():
    return{"status": "ok"}