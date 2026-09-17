import os
import sys
from typing import Optional, Union, Dict, Any

# Ensure backend directory is in sys.path regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from discord_notifier import notify_discord
from models import ResearchContext
from researcher import (
    run_research,
    run_chat,
    plan_research,
    approve_plan,
    get_research_state,
    get_session_claims,
    get_session_gaps,
    get_session_sufficiency,
)

load_dotenv()

app = FastAPI(title="SIH Research Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResearchRequest(BaseModel):
    topic: Optional[str] = None
    objective: Optional[str] = None
    context: Optional[Union[ResearchContext, Dict[str, Any], str]] = None
    purpose: Optional[str] = None
    session_id: str = "default"


@app.post("/research")
async def research(request: ResearchRequest):
    objective = request.objective or request.topic
    if not objective:
        raise HTTPException(status_code=400, detail="Either 'objective' or 'topic' is required.")

    result = await run_research(
        topic=objective,
        session_id=request.session_id,
        context=request.context,
        purpose=request.purpose,
    )
    await notify_discord(objective, request.session_id)
    return {"report": result}


class PlanRequest(BaseModel):
    topic: Optional[str] = None
    objective: Optional[str] = None
    context: Optional[Union[ResearchContext, Dict[str, Any], str]] = None
    purpose: Optional[str] = None
    session_id: str = "default"


@app.post("/plan")
async def create_plan(request: PlanRequest):
    objective = request.objective or request.topic
    if not objective:
        raise HTTPException(status_code=400, detail="Either 'objective' or 'topic' is required.")

    plan = await plan_research(
        objective=objective,
        session_id=request.session_id,
        context=request.context,
        purpose=request.purpose,
    )
    dump_fn = getattr(plan, "model_dump", None) or getattr(plan, "dict")
    return {"plan": dump_fn()}


class ApprovePlanRequest(BaseModel):
    session_id: str = "default"


@app.post("/plan/approve")
async def approve(request: ApprovePlanRequest):
    plan = approve_plan(request.session_id)
    if not plan:
        raise HTTPException(status_code=404, detail=f"No active research plan found for session: {request.session_id}")
    dump_fn = getattr(plan, "model_dump", None) or getattr(plan, "dict")
    return {"status": "approved", "plan": dump_fn()}


@app.get("/claims/{session_id}")
def get_claims(session_id: str):
    claims = get_session_claims(session_id)
    serialized = [
        c.model_dump() if hasattr(c, "model_dump") else c.dict()
        for c in claims
    ]
    return {"session_id": session_id, "claims": serialized}


@app.get("/claims/{session_id}/provenance")
def get_claims_provenance(session_id: str):
    state = get_research_state(session_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"No active research session found for: {session_id}")
    provenance_list = [
        state.get_claim_provenance(c.claim_id)
        for c in state.claims
    ]
    return {"session_id": session_id, "claims_provenance": provenance_list}


@app.get("/claims/{session_id}/{claim_id}/provenance")
def get_single_claim_provenance(session_id: str, claim_id: int):
    state = get_research_state(session_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"No active research session found for: {session_id}")
    prov = state.get_claim_provenance(claim_id)
    if not prov:
        raise HTTPException(status_code=404, detail=f"Claim {claim_id} not found in session: {session_id}")
    return prov


@app.get("/gaps/{session_id}")
def get_gaps(session_id: str):
    gaps = get_session_gaps(session_id)
    serialized = [
        g.model_dump() if hasattr(g, "model_dump") else g.dict()
        for g in gaps
    ]
    return {"session_id": session_id, "gaps": serialized}


@app.get("/sufficiency/{session_id}")
def get_sufficiency(session_id: str):
    sufficiency = get_session_sufficiency(session_id)
    if not sufficiency:
        return {"session_id": session_id, "sufficiency": None}
    dump_fn = getattr(sufficiency, "model_dump", None) or getattr(sufficiency, "dict")
    return {"session_id": session_id, "sufficiency": dump_fn()}


@app.get("/telemetry/{session_id}")
def get_telemetry(session_id: str):
    state = get_research_state(session_id)
    if not state or not state.telemetry:
        return {"session_id": session_id, "telemetry": None}
    dump_fn = getattr(state.telemetry, "model_dump", None) or getattr(state.telemetry, "dict")
    return {"session_id": session_id, "telemetry": dump_fn()}


@app.get("/health")
def health_check():
    return {"status": "ok"}


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


@app.post("/chat")
async def chat(request: ChatRequest):
    response = await run_chat(request.message, request.session_id)
    return {"response": response}