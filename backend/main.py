import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

load_dotenv()

app = FastAPI()

class ResearchRequest(BaseModel):
    topic: str

@app.post("/research")
def research(request: ResearchRequest):
    return {"received_topic": request.topic}

@app.get("/health")
def health_check():
    return{"status": "ok"}