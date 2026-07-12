import httpx
from bs4 import BeautifulSoup

MAX_REASONABLE_TAGS = 10  # postings with more tags than this are likely spam/tag-stuffing

def clean_description(raw_html: str) -> str:
    """Strip HTML tags from the description so the LLM stages don't waste
    tokens on <strong>, <ul><li>, <br/> etc."""
    soup = BeautifulSoup(raw_html, "html.parser")
    return soup.get_text(separator=" ", strip=True)


def fetch_jobs(tag: str = "machine-learning"):
    url = f"https://remoteok.com/api?tags={tag}"

    try:
        response = httpx.get(url, headers={"User-Agent": "research-assistant/0.1"})
        response.raise_for_status()

    except httpx.ConnectError:
        return {"error": "Could not reach RemoteOK API."}
    except httpx.TimeoutException:
        return {"error": "RemoteOK API timed out. Try again."}
    except httpx.HTTPStatusError as e:
        print(e.response.text)  # temporary — shows the real error
        return {"error": f"RemoteOK API returned error: {e.response.status_code}"}

    parsed_response = response.json()

    results = []

    for item in parsed_response:
        # Skip the legal/metadata object — it has a "legal" key, real jobs don't
        if "legal" in item:
            continue

        tags = item.get("tags", [])

        clean_job = {
            "title": item["position"],
            "company": item["company"],
            "description": clean_description(item["description"]),
            "tags": tags,
            "tag_count": len(tags),
            "likely_spam": len(tags) > MAX_REASONABLE_TAGS,
            "location": item["location"],
            "salary_min": item["salary_min"],
            "salary_max": item["salary_max"],
            "salary_disclosed": item["salary_min"] > 0 or item["salary_max"] > 0,
            "posted_at": item["date"],
            "apply_url": item["apply_url"],
            "source": "RemoteOK",
        }
        results.append(clean_job)

    return results


if __name__ == "__main__":
    results = fetch_jobs("machine-learning")
    for r in results:
        print(r["title"], "-", r["company"], "- spam flag:", r["likely_spam"])