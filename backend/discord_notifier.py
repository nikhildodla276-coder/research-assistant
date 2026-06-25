import httpx
import os
from dotenv import load_dotenv

load_dotenv()

async def notify_discord(topic: str, session_id: str):
    url = os.getenv("DISCORD_WEBHOOK_URL")
    
    if not url:
        print("DISCORD_WEBHOOK_URL not set in .env")
        return
    
    payload = {
        "content": f"**Research Complete**\n**Topic:** {topic}\n**Session:** {session_id}"
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload)
        print(f"Discord notification sent: {response.status_code}")