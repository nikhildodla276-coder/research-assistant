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
Built `researcher.py` with `run_research` function integrating `ChatGroq` (llama-3.3-70b-versatile) and `TavilySearch` (max 5 results). The function fetches live we