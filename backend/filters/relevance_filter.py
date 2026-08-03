import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

BATCH_SIZE = 25

def build_prompt(batch: list) -> str:
    """Builds one prompt containing multiple jobs, asking for one verdict per job."""
    listings_text = ""
    for i, job in enumerate(batch):
        listings_text += f"\n{i}. Title: {job['title']} | Company: {job['company']} | Tags: {', '.join(job['tags'])}\n"

    return f"""You are screening job listings for a candidate who wants a role at a 
SMALL (5-20 person), remote-native, funded AI/LLM-integration engineering startup.

For each listing below, answer only "relevant", "not-relevant", or "unsure".
Mark "not-relevant" for large/known big companies, non-engineering gig work 
(data labeling, AI training/evaluation contractor roles), or unrelated fields.
Mark "unsure" only if you genuinely cannot tell from the title/company/tags alone.

Listings:
{listings_text}

Respond ONLY with a JSON array of {len(batch)} strings, in order, nothing else.
Example: ["relevant", "not-relevant", "unsure"]
"""

def filter_relevant_jobs(jobs: list) -> list:
    """Takes cleaned job dicts, returns them with a 'relevance_verdict' field added."""
    results = []

    for i in range(0, len(jobs), BATCH_SIZE):
        batch = jobs[i:i + BATCH_SIZE]
        prompt = build_prompt(batch)

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
        )

        raw_reply = response.choices[0].message.content

        
        try:
            verdicts = json.loads(raw_reply)
            if len(verdicts) != len(batch):
                print(f"Expected {len(batch)} verdicts, got {len(verdicts)}. Raw reply:", raw_reply)
                verdicts = ["unsure"] * len(batch)
        except json.JSONDecodeError:
            print("Model did not return valid JSON:", raw_reply)
            verdicts = ["unsure"] * len(batch)


        VALID_VERDICTS = {"relevant", "not-relevant", "unsure"}

        for job, verdict in zip(batch, verdicts):
            if verdict not in VALID_VERDICTS:
                verdict = "unsure"
            job["relevance_verdict"] = verdict
            results.append(job)

    return results

if __name__ == "__main__":
    from fetchers.remoteok_fetcher import fetch_jobs

    jobs = fetch_jobs("backend")
    if isinstance(jobs, dict) and "error" in jobs:
        print("Fetch failed:", jobs["error"])
    else:
        filtered = filter_relevant_jobs(jobs)
        for job in filtered:
            print(job["relevance_verdict"], "-", job["title"], "-", job["company"])