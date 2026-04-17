"""
mcp_server.py — Standalone MCP (Model Context Protocol) server for web search.

Exposes a single tool 'web_search' via stdio transport using FastMCP.
The orchestrator spawns this as a subprocess and calls the tool via the
MCP client protocol. Can also be plugged into Claude Desktop, Cursor,
or any other MCP-compatible client independently.
"""

from mcp.server.fastmcp import FastMCP
from ddgs import DDGS
import sys
mcp = FastMCP("WebSearch")

@mcp.tool()
def web_search(query: str) -> str:
    """
    Search the web using DuckDuckGo and return the top 5 results.

    Each result includes the page title, a text snippet, and the source URL.
    Called by the LangGraph orchestrator for off-topic (non-academic) questions.
    Returns a formatted string ready to be passed as context to the LLM.
    """
    ddgs = DDGS()
    results = list(ddgs.text(query, max_results=5))
    if not results:
        return "No results found."
    output = []
    for r in results:
        output.append(f"Title: {r['title']}\nSnippet: {r['body']}\nURL: {r['href']}\n")
    return "\n".join(output)

if __name__ == "__main__":
    mcp.run(transport="stdio")
