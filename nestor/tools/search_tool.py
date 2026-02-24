"""Web search and page-fetching tools for Nestor."""

from __future__ import annotations

import logging
import re
from typing import Any
from urllib.parse import unquote

import httpx
from bs4 import BeautifulSoup

from nestor.tools import BaseTool

logger = logging.getLogger(__name__)

__all__ = ["WebSearchTool", "FetchWebPageTool"]

_SEARCH_URL = "https://html.duckduckgo.com/html/"
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
_MAX_PAGE_TEXT = 4000
_HTTP_TIMEOUT = 20.0


# ---------------------------------------------------------------------------
# Web search
# ---------------------------------------------------------------------------

class WebSearchTool(BaseTool):
    """Search the web via DuckDuckGo HTML endpoint."""

    name = "web_search"
    description = (
        "Search the web using DuckDuckGo and return a list of result titles, "
        "URLs, and snippets."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query.",
            },
            "num_results": {
                "type": "integer",
                "description": "Maximum number of results to return (default 5).",
            },
        },
        "required": ["query"],
    }

    async def execute(self, **kwargs: Any) -> str:
        query: str = kwargs["query"]
        num_results: int = kwargs.get("num_results", 5)

        try:
            async with httpx.AsyncClient(
                headers={"User-Agent": _USER_AGENT},
                timeout=_HTTP_TIMEOUT,
                follow_redirects=True,
            ) as client:
                resp = await client.post(_SEARCH_URL, data={"q": query})
                resp.raise_for_status()

            soup = BeautifulSoup(resp.text, "html.parser")
            result_links = soup.select("a.result__a")

            results: list[dict[str, str]] = []
            for link in result_links:
                if len(results) >= num_results:
                    break

                title = link.get_text(strip=True)
                raw_href = link.get("href", "")
                url = _extract_ddg_url(raw_href)
                if not url:
                    continue

                # Snippet lives in a sibling element.
                snippet_tag = link.find_parent("div", class_="result")
                snippet = ""
                if snippet_tag:
                    snippet_el = snippet_tag.select_one(".result__snippet")
                    if snippet_el:
                        snippet = snippet_el.get_text(strip=True)

                results.append({"title": title, "url": url, "snippet": snippet})

            if not results:
                return f'No results found for "{query}".'

            lines: list[str] = [f'Search results for "{query}":\n']
            for i, r in enumerate(results, 1):
                lines.append(f"{i}. {r['title']}")
                lines.append(f"   {r['url']}")
                if r["snippet"]:
                    lines.append(f"   {r['snippet']}")
                lines.append("")
            return "\n".join(lines).rstrip()
        except Exception as exc:
            logger.exception("Web search failed")
            return f"Error performing web search: {exc}"


def _extract_ddg_url(href: str) -> str | None:
    """Extract the real URL from a DuckDuckGo redirect link."""
    # DDG wraps links like //duckduckgo.com/l/?uddg=<encoded>&...
    match = re.search(r"uddg=([^&]+)", href)
    if match:
        return unquote(match.group(1))
    # Sometimes the href is a direct URL.
    if href.startswith("http"):
        return href
    return None


# ---------------------------------------------------------------------------
# Fetch web page
# ---------------------------------------------------------------------------

class FetchWebPageTool(BaseTool):
    """Fetch a URL and extract its main text content."""

    name = "fetch_web_page"
    description = (
        "Fetch a web page and extract its main text content. "
        "Returns up to 4 000 characters of extracted text."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to fetch.",
            },
        },
        "required": ["url"],
    }

    async def execute(self, **kwargs: Any) -> str:
        url: str = kwargs["url"]

        try:
            async with httpx.AsyncClient(
                headers={"User-Agent": _USER_AGENT},
                timeout=_HTTP_TIMEOUT,
                follow_redirects=True,
            ) as client:
                resp = await client.get(url)
                resp.raise_for_status()

            content_type = resp.headers.get("content-type", "")
            if "html" not in content_type and "text" not in content_type:
                return f"The URL returned non-text content ({content_type}). Cannot extract text."

            text = _extract_page_text(resp.text)
            if not text.strip():
                return "The page was fetched but no meaningful text content could be extracted."

            if len(text) > _MAX_PAGE_TEXT:
                text = text[:_MAX_PAGE_TEXT] + "\n\n[...truncated]"

            return f"Content from {url}:\n\n{text}"
        except httpx.HTTPStatusError as exc:
            return f"Error fetching {url}: HTTP {exc.response.status_code}"
        except Exception as exc:
            logger.exception("Failed to fetch web page %s", url)
            return f"Error fetching web page: {exc}"


def _extract_page_text(html: str) -> str:
    """Extract readable text from raw HTML.

    Strips scripts, styles, and navigation elements, then returns the
    remaining visible text cleaned of excessive whitespace.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove non-content elements.
    for tag in soup.find_all(["script", "style", "nav", "header", "footer", "noscript", "svg"]):
        tag.decompose()

    # Prefer <main> or <article> if present.
    main = soup.find("main") or soup.find("article")
    target = main if main else soup.body if soup.body else soup

    text = target.get_text(separator="\n")

    # Collapse whitespace.
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)
