import os

from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq

load_dotenv()

store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
chain = RunnableWithMessageHistory(llm, get_session_history)
tool = TavilySearch(max_results=5)

def run_research(topic: str, session_id: str = "default") -> str:
    results = tool.invoke(topic)
    prompt = f"Research topic: {topic}\n\nSearch results: {str(results)}\n\nWrite a clean structured report."
    response = chain.invoke(
        HumanMessage(content=prompt),
        config={"configurable": {"session_id": session_id}}
    )
    return response.content

def run_chat(message: str, session_id: str = "default") -> str:
    response = chain.invoke(
        HumanMessage(content=message),
        config={"configurable": {"session_id": session_id}}
    )
    return response.content