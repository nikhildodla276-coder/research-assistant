# sources.py
# Central config for trusted domains, organized by topic.
# Used by docs_fetcher.py (via Tavily's include_domains) and later
# by researcher.py to decide which domains are relevant to a query.
# Data only — no logic, no fetching happens here.

TRUSTED_SOURCES = {
    "langchain": [
        "python.langchain.com",
        "api.python.langchain.com",
    ],
    "langgraph": [
        "langchain-ai.github.io",
    ],
    "fastapi": [
        "fastapi.tiangolo.com",
    ],
    "groq": [
        "console.groq.com",
    ],
    "tavily": [
        "docs.tavily.com",
    ],
    "langsmith": [
        "docs.smith.langchain.com",
    ],
    "python": [
        "docs.python.org",
    ],
}