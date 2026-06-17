import os
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage

memory = ConversationBufferMemory

def run_research(topic:str) -> str:
    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
    tool = TavilySearchResults(max_results=5)
    results = tool.invoke(topic)
    prompt = f"Research topic:{topic}\n\nSearch results:{str(results)}\n\nWrite a clean structured research report based on these results."
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content
