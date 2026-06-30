import httpx
from urllib.parse import urlencode
```python
def fetch_hn(query:str, tags:str="story", min_points:int=20):
    dictionary = {
        "query": query, "tags": tags, "numericFilters": f"points>{min_points}", "hitsPerPage": 20
    }
    url = "https://hn.algolia.com/api/v1/search" + "?" + urlencode(dictionary)
    ```python
  try:
        response= httpx.get(url)
        response.raise_for_status()

    except httpx.ConnectError:
        return{"error": "Could not reach HN API."}
    except httpx.TimeoutException:
        return{"error": "HN API timed out. Try again."}
    except httpx.HTTPStatusError as e:
        return{"error": f"HN API returned error: {e.response.status_code}"}
```
