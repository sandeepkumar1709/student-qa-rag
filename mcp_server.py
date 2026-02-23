from mcp.server.fastmcp import FastMCP
from ddgs import DDGS
import sys
mcp = FastMCP("WebSearch")

@mcp.tool()
def web_search(query: str) -> str:
    """Search the web using DuckDuckGo and return top results"""
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
