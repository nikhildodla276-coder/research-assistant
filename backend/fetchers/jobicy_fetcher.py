import httpx

MAX_REASONABLE_TAGS = 10  # kept for schema parity with jobs_fetcher.py; Himalayas has no tags field, always 0/False


def fetch_himalayas(keyword: str = "machine learning", limit: int = 20):
    url = "https://himalayas.app/jobs/api"
    params = {"limit": limit, "offset": 0}

    try:
        response = httpx.get(url, params=params, headers={"User-Agent": "research-assistant/0.1"})
        response.raise_for_status()

    except httpx.ConnectError:
        return {"error": "Could not reach Himalayas API."}
    except httpx.TimeoutException:
        return {"error": "Himalayas API timed out. Try again."}
    except httpx.HTTPStatusError as e:
        print(e.response.text)  # temporary — shows the real error
        return {"error": f"Himalayas API returned error: {e.response.status_code}"}

    parsed_response = response.json()

    results = []

    for item in parsed_response.get("jobs", []):
        min_salary = item.get("minSalary") or 0
        max_salary = item.get("maxSalary") or 0

        clean_job = {
            "title": item["title"],
            "company": item["companyName"],
            "description": item.get("excerpt", ""),
            "tags": [],  # Himalayas browse endpoint doesn't return a tags/skills list
            "tag_count": 0,
            "likely_spam": False,  # no tag data to base this on for Himalayas
            "location": item.get("locationRestrictions") or "Not specified",
            "salary_min": min_salary,
            "salary_max": max_salary,
            "salary_disclosed": min_salary > 0 or max_salary > 0,
            "posted_at": parsed_response.get("updatedAt"),  # browse endpoint gives feed-level timestamp, not per-job
            "apply_url": f"https://himalayas.app/companies/{item.get('companySlug', '')}/jobs/{item.get('slug', '')}",
            "source": "Himalayas",
        }
        results.append(clean_job)

    # Client-side keyword filter since the browse endpoint doesn't accept a query param
    if keyword:
        keyword_lower = keyword.lower()
        results = [
            r for r in results
            if keyword_lower in r["title"].lower() or keyword_lower in r["description"].lower()
        ]

    return results


if __name__ == "__main__":
    results = fetch_himalayas("machine learning")

    if isinstance(results, dict) and "error" in results:
        print("Fetch failed:", results["error"])
    else:
        print(f"Got {len(results)} results after keyword filter")
        for r in results:
            print(r["title"], "-", r["company"], "-", r["location"])