"""
onet_fetcher.py

Fetches structured occupational data from O*NET Web Services.
Free, government-run REST API — requires a free API key from
https://services.onetcenter.org/

WHY THIS FILE IS SIMPLE: O*NET already publishes structured JSON, same
category as RemoteOK's API. No search, no source-scoring, no LLM
extraction needed for the raw fetch itself.

WHAT THIS ANSWERS: "what does this occupation involve, what skills/tools
does it require, what's its outlook" — NOT "why is there a hiring surge
right now" (that's the separate, harder BLS/Tavily fetcher, built after
this one is tested).
"""

import httpx
import asyncio
import os


# ---- constants ----

ONET_BASE_URL = "https://services.onetcenter.org/ws/online/occupations"

# O*NET uses HTTP Basic Auth: your registered email as username,
# the API key they email you as password. Store both as env vars —
# never hardcode credentials in the file itself.
ONET_USERNAME = os.environ.get("ONET_USERNAME")
ONET_PASSWORD = os.environ.get("ONET_PASSWORD")

# O*NET-SOC codes for roles relevant to you. You find these once by
# searching O*NET OnLine's website manually (onetonline.org), then
# hardcode them here — the codes are stable, so no need to search
# programmatically every time.
TARGET_SOC_CODES = {
    "software_developers": "15-1252.00",
    "data_scientists": "15-2051.00",
    "web_developers": "15-1254.00",
}


# ---- core fetch function ----

async def fetch_occupation_details(client: httpx.AsyncClient, soc_code: str) -> dict:
    """
    Fetches one occupation's full report from O*NET.

    Why async httpx: matches your discord_notifier.py pattern, and lets
    you fetch multiple SOC codes concurrently later instead of one-by-one.

    Why we pass in the client rather than creating one per call: creating
    a new httpx.AsyncClient for every request is wasteful — one client
    should be reused across all calls in a single run, same idea as
    reusing one DB connection instead of opening a new one per query.
    """
    url = f"{ONET_BASE_URL}/{soc_code}/summary"

    response = await client.get(
        url,
        auth=(ONET_USERNAME, ONET_PASSWORD),
        headers={"Accept": "application/json"},
    )

    # Real error handling, not just assuming success:
    # O*NET returns 401 for bad credentials, 404 for an unknown SOC code.
    # We raise here rather than silently returning empty data, because
    # silent failures are exactly what "filter-before-store" is designed
    # to prevent — better to crash loudly during testing than store junk.
    response.raise_for_status()

    return response.json()


def normalize_onet_record(raw: dict, soc_code: str, role_key: str) -> dict:
    """
    Converts O*NET's raw JSON into your project's internal schema.

    Why normalize at all: O*NET's raw JSON has its own field names and
    nesting (e.g. it may nest tasks under different keys depending on
    endpoint version). Your project's storage and LLM-analysis stage
    should never have to know O*NET's specific quirks — normalization
    is the same principle as job_fetchers.py stripping RemoteOK's HTML
    and skipping its metadata row before storage.

    The "source" field is not optional. O*NET's free API access requires
    attribution — every record must carry a pointer back to O*NET, same
    attribution-preservation rule already used for job listings.
    """
    return {
        "role_key": role_key,
        "soc_code": soc_code,
        "title": raw.get("title"),
        "description": raw.get("description"),
        # .get() with a default avoids a KeyError crash if O*NET's
        # response shape doesn't include a field for this occupation —
        # some fields are genuinely optional per O*NET's own docs.
        "sample_job_titles": raw.get("sample_of_reported_job_titles", []),
        "source": "O*NET Web Services",
        "source_url": f"https://www.onetonline.org/link/summary/{soc_code}",
        "fetched_via": "onet_fetcher.py",
    }


async def run_onet_fetch(role_keys: list[str]) -> list[dict]:
    """
    Orchestrator: fetches + normalizes every role in role_keys.

    Why one shared client + asyncio.gather: fetching each SOC code is an
    independent network call, so we run them concurrently rather than
    awaiting one at a time — same async-for-speed principle already
    used in your job_fetchers.py / discord_notifier.py conversion.
    """
    results = []

    async with httpx.AsyncClient(timeout=30.0) as client:
        tasks = []
        for role_key in role_keys:
            soc_code = TARGET_SOC_CODES.get(role_key)
            if soc_code is None:
                # Same "never silently fake groundedness" principle as
                # the router design: if we don't have a code for this
                # role, say so clearly instead of skipping silently.
                print(f"[onet_fetcher] No SOC code mapped for '{role_key}' — skipping.")
                continue
            tasks.append(fetch_occupation_details(client, soc_code))

        raw_responses = await asyncio.gather(*tasks, return_exceptions=True)

        for role_key, raw in zip(
            [k for k in role_keys if k in TARGET_SOC_CODES], raw_responses
        ):
            if isinstance(raw, Exception):
                # A single failed fetch shouldn't crash the whole batch —
                # log it and continue, same resilience pattern you'd want
                # in job_fetchers.py if one RemoteOK listing was malformed.
                print(f"[onet_fetcher] Failed to fetch '{role_key}': {raw}")
                continue
            normalized = normalize_onet_record(raw, TARGET_SOC_CODES[role_key], role_key)
            results.append(normalized)

    return results


# ---- test block ----

if __name__ == "__main__":
    # Run against ONE real role first. Don't test all three at once —
    # if something breaks, you want to know it's a real O*NET/auth issue,
    # not guess whether it's role-specific.
    async def main():
        results = await run_onet_fetch(["software_developers"])
        for r in results:
            print(r)

    asyncio.run(main())