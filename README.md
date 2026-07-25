# AI Research Assistant

A full-stack AI-powered research tool built with FastAPI and LangChain. Given a topic, it searches the web in real time, generates a structured research report using an LLM, supports follow-up conversation with memory, and sends a Discord notification on research completion.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI, Python 3.13 |
| LLM | Groq API — `llama-3.3-70b-versatile` |
| Web Search | Tavily Search (`langchain_tavily`) |
| Memory | `InMemoryChatMessageHistory`, `RunnableWithMessageHistory` |
| HTTP Client | httpx (async) |
| Notifications | Discord Webhook |
| Frontend | HTML, CSS, JavaScript, marked.js |
| Environment | python-dotenv |

---

## Project Structure

```
research-assistant/
├── backend/
│   ├── main.py                  # FastAPI app, routes, middleware
│   ├── researcher.py            # LangChain pipeline — run_research, run_chat
│   ├── discord_notifier.py      # Discord webhook notification
│   ├── .env                     # API keys and webhook URL (not committed)
│   └── venv/                    # Virtual environment
├── index.html                   # Frontend UI
├── style.css                    # Dark theme styles
├── script.js                    # Frontend logic, session management
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Features

- **Real-time web research** — Tavily searches the web and feeds results to the LLM
- **Structured report generation** — LLM produces a clean, markdown-formatted research report
- **Persistent conversation memory** — Follow-up questions maintain context per session using `InMemoryChatMessageHistory`
- **Session isolation** — Each browser session gets a unique `session_id` generated via `Date.now()`
- **Markdown rendering** — Reports and chat responses rendered via `marked.js`
- **Discord notifications** — Captain Hook webhook posts topic and session ID to `#general` on research completion
- **Dark theme frontend** — Three-file frontend with research panel and persistent chat panel

---

## API Endpoints

| Method | Route | Description |
|---|---|---|
| GET | `/health` | Health check — returns `{"status": "ok"}` |
| POST | `/research` | Accepts `topic` and `session_id`, returns research report |
| POST | `/chat` | Accepts `message` and `session_id`, returns LLM response with memory |

### Request Models

**ResearchRequest**
```json
{
  "topic": "quantum computing",
  "session_id": "1234567890"
}
```

**ChatRequest**
```json
{
  "message": "Explain the key findings in simpler terms",
  "session_id": "1234567890"
}
```

---

## Setup and Installation

### Prerequisites

- Python 3.13
- Groq API key — [console.groq.com](https://console.groq.com)
- Tavily API key — [tavily.com](https://tavily.com)
- Discord server with a webhook URL

### Steps

**1. Clone the repository**
```bash
git clone https://github.com/nikhildodla276-coder/research-assistant.git
cd research-assistant
```

**2. Create and activate virtual environment**
```bash
cd backend
python -m venv venv
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Configure environment variables**

Create a `.env` file inside the `backend/` folder:
```
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
DISCORD_WEBHOOK_URL=your_discord_webhook_url
```

**5. Run the server**
```bash
uvicorn main:app --reload
```

**6. Open the frontend**

Open `index.html` directly in your browser.

---

## Development Phases

### Phase 1 — FastAPI Foundation
Established the FastAPI project structure with a health check endpoint (`GET /health`) and a research route (`POST /research`). Added Pydantic request validation with `ResearchRequest` model. Verified via Swagger UI at `/docs`.

### Phase 2 — LangChain Research Pipeline
Built `researcher.py` with `run_research` function integrating `ChatGroq` (llama-3.3-70b-versatile) and `TavilySearch` (max 5 results). The function fetches live web results, builds a structured prompt, and returns an LLM-generated report.

### Phase 3 — Conversation Memory
Implemented `InMemoryChatMessageHistory` with `RunnableWithMessageHistory` for per-session memory. Added `run_chat` function for follow-up questions that skip Tavily and query the LLM directly with full conversation history. Memory is stored in a `store` dict keyed by `session_id`.

### Phase 4 — Frontend
Built a three-file dark-theme frontend (`index.html`, `style.css`, `script.js`). Features a research panel for report display and a persistent chat panel for follow-up conversation. Markdown rendering via `marked.js`. Unique `session_id` generated per page load using `Date.now()`. Added `CORSMiddleware` to `main.py`.

### Phase 5 — Discord Webhook Notifications
Created `discord_notifier.py` with an async `notify_discord(topic, session_id)` function using `httpx.AsyncClient`. After every successful research completion, a formatted notification is POSTed to a Discord channel via webhook. Integrated into the `/research` route in `main.py`.

---

## Environment Variables

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Groq API key for LLM access |
| `TAVILY_API_KEY` | Tavily API key for web search |
| `DISCORD_WEBHOOK_URL` | Discord channel webhook URL for notifications |

---

## Author

**Nikhil Dodla**
BTech CSE AIML — Kalinga University, Raipur
[GitHub](https://github.com/nikhildodla276-coder)

currently working on building a concrete plan on how to modify my present working research assistant project to a real world useful tool which helps my production more siginificantly by replacing all the boring tasks to be done by AI.

## What We Are Building — Final Confirmed Plan

# Research Assistant — What, Why, How

## What this is
A personal AI research tool. It searches the web, summarizes findings 
with an LLM, supports follow-up Q&A, and preserves source attribution 
on every claim it stores. Two planned modes:
- Research mode — on-demand queries, built and working
- Daily Brief mode — not yet designed

## Why it exists
Built to solve a real, specific problem in my own career search: 
finding a full-time remote AI/backend engineering role at a small 
(5-20 person), remote-native, USD/GBP/EUR-funded startup doing real 
AI/LLM-integration work — right after graduation, roughly two years 
from now.

The real hiring mechanism for this company type is founder networks, 
direct outreach, and referrals — not cold platform applications. But 
before any of that outreach can be credible, I need to know, at 
volume: what roles/skills the market actually wants, and which 
specific companies are legitimate, funded, and doing the work they 
claim to do. That's what this project does — it is a market-
intelligence and company-research tool, not a job-application tool.

## How it helps me
1. Fetches job postings at volume from multiple platforms
2. Researches each relevant posting's company completely — funding, 
   team size, remote-native evidence, real engineering work — 
   validating or disproving fit against my four-filter criteria
3. Analyzes patterns across everything validated — common skills, 
   stacks, and problems being solved
4. Feeds two parallel tracks going forward: what skills to actually 
   learn, and which specific companies are worth targeting for 
   contribution/DMs/cold outreach

## Future use case
Once mature, this becomes the tool I check regularly (Daily Brief 
mode) the way I'd check news — surfacing new relevant postings, 
newly-discovered legitimate companies, and shifts in what skills are 
in demand, right up to graduation.

## Current status
Research mode: built (FastAPI + LangChain + Groq + Tavily pipeline, 
conversation memory, frontend UI). 
Daily Brief mode: not yet designed.
Fetchers: hn_fetcher.py and job_fetchers.py (RemoteOK) done and 
tested. Himalayas, Jobicy, Arbeitnow, onet_fetcher.py designed or 
in progress, not yet complete.

## Architecture
RemoteOK is the first fetcher to build, understood honestly as a broad-but-large remote board requiring real tag/keyword filtering to surface AI-relevant roles — not an AI-specific source.
Wellfound, LinkedIn (indirect via Tavily only), and later Arbeitnow/Jobicy/Himalayas remain the next platforms, added when a specific gap shows up, not preemptively.
Career-context/"why" source category confirmed as necessary, domains not yet chosen.
Dedicated new-skill-learning deferred until jobs+career data confirms priorities; learning-through-building continues implicitly and isn't paused.
Daily Brief mode identified as the eventual home for recurring "what am I lacking, what's shifted" questions — not built yet, Phase 2 per original blueprint.

- Confirmed tags=machine-learning query returns real, mostly-legitimate ML/AI postings
- Confirmed spam flag (tag_count > 10) correctly stays False on clean, non-tag-stuffed results
- Identified real data-quality gaps to address: non-role postings (e.g. generic "apply anyway"
  listings), duplicate postings from same company, and adjacent-but-not-target roles
  (management/architecture) passing the tag filter
- Fixed __main__ block to handle fetch_jobs() error-dict return path instead of assuming
  success and crashing on TypeError

## Project Documentation
Detailed build history, architecture decisions, and fetcher notes 
now live in a structured docs folder (not in this README) — kept 
in personal notes, updated as the project grows.

added jobicy_fetcher to backend fetchers
added onet_fetcher to backend fetchers