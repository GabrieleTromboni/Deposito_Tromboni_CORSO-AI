"""Utility functions: somma e ricerca web con DuckDuckGo.

Contiene una funzione elementare `somma(a, b)` e una funzione
`search_web_ddgs` che effettua ricerche testuali mediante ddgs.
"""

from duckduckgo_search import DDGS

__all__ = ["somma", "search_web_ddgs"]

# Funzione che fa la somma di due numeri interi

def somma(a: int, b: int) -> int:
    """Calcola la somma di due numeri interi.

    Args:
        a (int): Il primo numero intero.
        b (int): Il secondo numero intero.

    Returns:
        int: La somma di a e b.
    """
    return a + b

# Funzione che fa la ricerca web usando DuckDuckGo

def search_web_ddgs(
    query: str,
    max_results: int = 10,
    *,
    region: str = "it-it",
    safesearch: str = "moderate",
    timelimit: str | None = None,
) -> list[dict[str, str]]:
    """Search the web using DuckDuckGo (ddgs) and return structured results.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return (best-effort).
        region: Region code, e.g., "wt-wt" (worldwide), "us-en", "it-it".
        safesearch: One of {"off", "moderate", "strict"}.
        timelimit: Optional time filter (e.g., "d", "w", "m", "y").
        (no timeout parameter; uses ddgs defaults).

    Returns:
        A list of dictionaries with keys: "title", "url", and "snippet".

    Raises:
        ValueError: If the query is empty or parameters are invalid.
    """

    if not isinstance(query, str):
        raise ValueError("query must be a string")
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("query cannot be empty or whitespace-only")

    if max_results <= 0:
        raise ValueError("max_results must be > 0")

    if safesearch not in {"off", "moderate", "strict"}:
        raise ValueError("safesearch must be one of {'off','moderate','strict'}")

    # Collect results with deduplication by URL while preserving order
    results: list[dict[str, str]] = []
    seen_urls: set[str] = set()

    # Use context manager to ensure proper session cleanup
    with DDGS(verify=False) as ddgs:
        for item in ddgs.text(
            normalized_query,
            max_results=max_results,
            region=region,
            safesearch=safesearch,
            timelimit=timelimit,
        ):
            # ddgs returns items like {"title", "href", "body", ...}
            title = (item.get("title") or "").strip()
            url = (item.get("href") or "").strip()
            snippet = (item.get("body") or "").strip()

            if not url or url in seen_urls:
                continue

            results.append({
                "title": title,
                "url": url,
                "snippet": snippet,
            })
            seen_urls.add(url)

            if len(results) >= max_results:
                break

    return results


if __name__ == "__main__":
    # Minimal runnable example
    DEMO_QUERY = "site:python.org typing"
    for rank, r in enumerate(search_web_ddgs(DEMO_QUERY, max_results=5, region="wt-wt"), start=1):
        print(f"{rank}. {r['title']}\n   {r['url']}\n   {r['snippet'][:160]}\n")
