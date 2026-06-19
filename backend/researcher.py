import os
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
memory = ConversationBufferMemory()
chain = ConversationChain(llm=llm, memory=memory, verbose=False)

def run_research(topic:str) -> str:
    tool = TavilySearchResults(max_results=5)
    results = tool.invoke(topic)
    prompt = f"Research topic:{topic}\n\nSearch results:{str(results)}\n\nWrite a clean structured research report based on these results."
    return chain.predict(input=prompt)