import httpx
from urllib.parse import urlencode

def fetch_hn(query:str, tags:str="story", min_points:int=20):
    url_parms = {
        "query": query, "tags": tags, "hitsPerPage": 20
    }
    url = "https://hn.algolia.com/api/v1/search" + "?" + urlencode(url_parms)

    try:
        response= httpx.get(url)
        response.raise_for_status()

    except httpx.ConnectError:
        return{"error": "Could not reach HN API."}
    except httpx.TimeoutException:
        return{"error": "HN API timed out. Try again."}
    except httpx.HTTPStatusError as e:
        print(e.response.text)  # temporary — shows the real error
        return {"error": f"HN API returned error: {e.response.status_code}"}
    
    parsed_response = response.json()

    results =[]

    for hit in parsed_response["hits"]:
        clean_hit = {
            "title": hit["title"],
            "url": hit["url"],
            "hn_discussion": f"https://news.ycombinator.com/item?id={hit['story_id']}",
            "author": hit["author"],
            "points": hit["points"],
            "num_comments": hit["num_comments"],
            "posted_at": hit["created_at"],
            "source": "Hacker News"
        }
        if hit["points"] >= min_points:
            results.append(clean_hit)

    return results


if __name__ == "__main__":
    results = fetch_hn("LangChain")
    for r in results:
        print(r)