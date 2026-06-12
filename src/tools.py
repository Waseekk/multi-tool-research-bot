"""
src/tools.py
============
Defines all 11 tools available to the agent and the initialize_tools() factory
that assembles them at startup.

Tool categories:
  - LangChain wrappers : ArXiv, Wikipedia, PubMed  (pre-built, just need wrappers)
  - Custom REST tools  : Semantic Scholar, OpenAlex (no official LangChain wrappers exist)
  - Web search         : Tavily (optional, paid), DuckDuckGo (free, no key)
  - Utility tools      : calculator, code_analyzer, weather_info, file_content_generator

To add a new tool:
  1. Define it with @tool here
  2. Add it to the list in initialize_tools()
  3. Add its name (as registered by LangChain) and description to the
     system prompt in context_aware_llm (nodes.py)
  4. Add keyword detection to get_task_optimized_llm (nodes.py) if it needs
     a specific temperature

Tool names as registered by LangChain (used in the system prompt):
  arxiv, wikipedia, pub_med, tavily_search_results_json,
  duckduckgo_search, calculator, code_analyzer, weather_info,
  file_content_generator, semantic_scholar_search, openalex_search
"""

import os
import json
import random
from datetime import datetime
from typing import List

from dotenv import load_dotenv
load_dotenv()

from langchain_core.tools import tool
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, PubmedQueryRun
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper, PubMedAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

# TavilySearchResults from langchain_community is the stable wrapper — prefer it.
# langchain_tavily.TavilySearch has a bug where newer tavily-python versions changed
# the response structure and it calls `.results` on an object that no longer has that
# attribute, causing a runtime crash on every web search tool call.
try:
    from langchain_community.tools.tavily_search import TavilySearchResults
    TAVILY_AVAILABLE = True
except ImportError:
    try:
        from langchain_tavily import TavilySearch
        TAVILY_AVAILABLE = True
    except ImportError:
        TAVILY_AVAILABLE = False
        print("Tavily not available. Install with: pip install langchain-community")


# ---------------------------------------------------------------------------
# Utility tools
# ---------------------------------------------------------------------------

@tool
def calculator(expression: str) -> str:
    """
    Perform mathematical calculations safely.

    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 3 * 4", "15/100 * 2500")

    Returns:
        Result of the calculation or an error message
    """
    # Whitelist of allowed characters instead of ast.literal_eval or sandboxed exec.
    # The LLM only sends simple arithmetic, so full expression parsing is unnecessary,
    # and this approach keeps the attack surface minimal.
    allowed_chars = set("0123456789+-*/.() ")
    if not all(c in allowed_chars for c in expression):
        return "Error: Invalid characters. Only numbers and basic operators (+, -, *, /, parentheses) are allowed."

    if "%" in expression:
        return "Error: Use decimal form instead of %. Example: '15/100 * 2500' instead of '15% * 2500'"

    try:
        result = eval(expression)   # safe: input is constrained to the whitelist above
        return f"Result: {result}"
    except Exception as e:
        return f"Error calculating '{expression}': {str(e)}"


@tool
def code_analyzer(code: str, language: str = "python") -> str:
    """
    Analyze code for basic syntax issues and return structural feedback.

    Args:
        code: Code snippet to analyze
        language: Programming language (default: "python")

    Returns:
        Analysis report as a formatted string
    """
    try:
        feedback = (
            f"Code Analysis for {language}:\n"
            f"- Lines of code: {len(code.splitlines())}\n"
            f"- Characters: {len(code)}\n"
            f"- Contains functions: {'def ' in code if language == 'python' else 'function ' in code}\n"
            f"- Contains classes: {'class ' in code if language == 'python' else False}\n"
            f"- Contains imports: {any(l.strip().startswith(('import ', 'from ')) for l in code.splitlines()) if language == 'python' else False}"
        )

        if language == "python":
            # compile() validates syntax without executing the code
            try:
                compile(code, "<string>", "exec")
                feedback += "\n- Syntax: Valid Python syntax"
            except SyntaxError as e:
                feedback += f"\n- Syntax Error: {str(e)}"

        return feedback
    except Exception as e:
        return f"Error analyzing code: {str(e)}"


@tool
def weather_info(location: str) -> str:
    """
    Get weather information for a location.

    Note: Returns simulated data. Replace the body of this function with a
    real weather API call (e.g. OpenWeatherMap) when ready for production.

    Args:
        location: City or location name

    Returns:
        Weather report string (currently demo/simulated data)
    """
    conditions = ["sunny", "cloudy", "rainy", "partly cloudy", "windy"]
    return (
        f"Weather for {location} (Demo Data):\n"
        f"- Temperature: {random.randint(15, 35)}C\n"
        f"- Condition: {random.choice(conditions).title()}\n"
        f"- Humidity: {random.randint(40, 80)}%\n"
        f"- Wind Speed: {random.randint(5, 25)} km/h\n"
        "Note: This is simulated data for demonstration purposes."
    )


@tool
def file_content_generator(file_type: str, content_description: str) -> str:
    """
    Generate sample file content for a given type and description.

    Args:
        file_type: One of 'csv', 'json', 'python', 'markdown'
        content_description: Human-readable description of what to include

    Returns:
        Generated file content as a string ready to copy/paste or save
    """
    try:
        if file_type.lower() == "csv":
            return (
                f"# Sample CSV for: {content_description}\n"
                "name,age,city,score\n"
                "Alice,25,New York,85\n"
                "Bob,30,London,92\n"
                "Charlie,35,Tokyo,78\n"
                "Diana,28,Paris,88"
            )
        elif file_type.lower() == "json":
            return json.dumps({
                "description": content_description,
                "data": [
                    {"id": 1, "name": "Item 1", "value": 100},
                    {"id": 2, "name": "Item 2", "value": 250},
                    {"id": 3, "name": "Item 3", "value": 175},
                ],
                "metadata": {"created": datetime.now().isoformat(), "version": "1.0"},
            }, indent=2)
        elif file_type.lower() == "python":
            return (
                f'"""{content_description}"""\n\n'
                "def main():\n"
                "    data = [1, 2, 3, 4, 5]\n"
                "    print(f'Result: {process_data(data)}')\n\n"
                "def process_data(data):\n"
                "    return sum(data)\n\n"
                'if __name__ == "__main__":\n'
                "    main()\n"
            )
        elif file_type.lower() == "markdown":
            return (
                f"# {content_description}\n\n"
                f"## Overview\nThis document covers {content_description.lower()}.\n\n"
                "## Key Points\n- Point 1\n- Point 2\n- Point 3\n\n"
                "## Code Example\n```python\ndef example():\n    return 'Hello'\n```\n"
            )
        else:
            return f"Sample content for {file_type}:\n{content_description}\n\nGenerated: {datetime.now()}"
    except Exception as e:
        return f"Error generating {file_type} content: {str(e)}"


# ---------------------------------------------------------------------------
# Custom research tools (no official LangChain wrappers exist for these APIs)
# ---------------------------------------------------------------------------

@tool
def semantic_scholar_search(query: str) -> str:
    """
    Search Semantic Scholar for academic papers across all scientific fields.
    Especially useful for citation counts and cross-field research impact.

    Uses the public Semantic Scholar Graph API — no API key required.

    Args:
        query: Search terms (e.g. "transformer attention mechanism")

    Returns:
        Formatted list of top 5 papers with title, authors, year, citation count, and abstract
    """
    import requests
    try:
        # limit removed from signature: LLMs pass it as a string, causing Groq schema validation
        # to reject the tool call before it reaches this code. Hardcoded to 5 instead.
        limit = 5
        resp = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={
                "query": query,
                "limit": min(limit, 10),
                "fields": "title,abstract,authors,year,citationCount,url",
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        if not data.get("data"):
            return f"No papers found for: {query}"

        results = []
        for i, paper in enumerate(data["data"], 1):
            authors = paper.get("authors", [])
            author_str = ", ".join(a.get("name", "") for a in authors[:3])
            if len(authors) > 3:
                author_str += " et al."
            abstract = (paper.get("abstract") or "")[:300]
            if len(paper.get("abstract") or "") > 300:
                abstract += "..."
            results.append(
                f"**{i}. {paper.get('title', 'Unknown')}**\n"
                f"- Authors: {author_str}\n"
                f"- Year: {paper.get('year', 'N/A')} | Citations: {paper.get('citationCount', 0)}\n"
                f"- URL: {paper.get('url', '')}\n"
                f"- Abstract: {abstract or 'N/A'}\n"
            )
        return f"## Semantic Scholar: '{query}'\n" + "\n".join(results)

    except requests.exceptions.Timeout:
        return "Error: Semantic Scholar API timed out."
    except requests.exceptions.RequestException as e:
        return f"Error: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def openalex_search(query: str) -> str:
    """
    Search OpenAlex for scholarly works across all academic disciplines.
    OpenAlex is a fully open, free catalog of global research output.

    No API key required. OpenAlex's terms ask for a contact email in User-Agent,
    which is included in the request headers below.

    Args:
        query: Search terms

    Returns:
        Formatted list of top 5 works with title, authors, year, citations, and open-access URL if available
    """
    import requests
    try:
        # limit removed from signature: same reason as semantic_scholar_search above
        limit = 5
        resp = requests.get(
            "https://api.openalex.org/works",
            params={
                "search": query,
                "per-page": min(limit, 10),
                "select": "title,authorships,publication_year,cited_by_count,open_access,doi",
            },
            # OpenAlex terms of service require a contact email in User-Agent for polite API usage
            headers={"User-Agent": "MultiToolResearchBot/1.0 (mailto:research@example.com)"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        if not data.get("results"):
            return f"No works found for: {query}"

        results = []
        for i, work in enumerate(data["results"], 1):
            authorships = work.get("authorships", [])
            author_str = ", ".join(
                a.get("author", {}).get("display_name", "Unknown") for a in authorships[:3]
            )
            if len(authorships) > 3:
                author_str += " et al."
            oa = work.get("open_access", {})
            oa_line = f"- Open Access URL: {oa.get('oa_url')}\n" if oa.get("oa_url") else ""
            results.append(
                f"**{i}. {work.get('title', 'Unknown')}**\n"
                f"- Authors: {author_str}\n"
                f"- Year: {work.get('publication_year', 'N/A')} | Citations: {work.get('cited_by_count', 0)}\n"
                f"- Access: {'Open Access' if oa.get('is_oa') else 'Closed Access'}\n"
                f"- DOI: {work.get('doi', 'N/A')}\n"
                f"{oa_line}"
            )
        return f"## OpenAlex: '{query}'\n" + "\n".join(results)

    except requests.exceptions.Timeout:
        return "Error: OpenAlex API timed out."
    except requests.exceptions.RequestException as e:
        return f"Error: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"


# ---------------------------------------------------------------------------
# Tool factory
# ---------------------------------------------------------------------------

def initialize_tools() -> List:
    """
    Instantiates and returns all tools in the order they are added to the list.

    Always available (no API key):
      arxiv, wikipedia, pub_med, duckduckgo_search,
      calculator, code_analyzer, weather_info, file_content_generator,
      semantic_scholar_search, openalex_search

    Conditionally available:
      tavily_search_results_json — only added when TAVILY_API_KEY env var is set

    Note on wrapper config:
      - Wikipedia: top_k_results=2, doc_content_chars_max=800
        Wikipedia articles are very long; larger values overflow the model context.
      - ArXiv: top_k_results=10, doc_content_chars_max=10000
        ArXiv abstracts are short; we can safely return more results.
      - PubMed: top_k_results=5, doc_content_chars_max=1500
        Requires `xmltodict` package (pip install xmltodict).

    Returns
    -------
    List of LangChain tool objects, ready to pass to llm.bind_tools() and ToolNode()
    """
    # Wikipedia returns long full-article text; limit results to avoid overflowing context
    wiki_tool = WikipediaQueryRun(
        api_wrapper=WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=800),
        description="Search Wikipedia for general knowledge, definitions, and factual information.",
    )
    arxiv_tool = ArxivQueryRun(
        api_wrapper=ArxivAPIWrapper(top_k_results=10, doc_content_chars_max=10000),
        description="Search academic papers on arXiv. Best for recent research papers and preprints.",
    )
    # PubmedQueryRun requires the `xmltodict` package to parse PubMed XML responses
    pubmed_tool = PubmedQueryRun(
        api_wrapper=PubMedAPIWrapper(top_k_results=5, doc_content_chars_max=1500),
        description="Search PubMed for medical, biomedical, life sciences, and healthcare research.",
    )

    tools_list = [arxiv_tool, wiki_tool, pubmed_tool]

    # Tavily is a paid service — only included when the API key is present.
    # Prefer TavilySearchResults (langchain_community) — stable and avoids the
    # '.results' attribute crash in langchain_tavily with newer tavily-python versions.
    if TAVILY_AVAILABLE and os.getenv("TAVILY_API_KEY"):
        try:
            try:
                # TavilySearchResults is the preferred stable wrapper
                tavily_tool = TavilySearchResults(
                    max_results=5,
                    description="Search the web for recent news and real-time information.",
                )
            except NameError:
                # TavilySearchResults not imported — fall back to new langchain_tavily package
                tavily_tool = TavilySearch(
                    max_results=5,
                    description="Search the web for recent news and real-time information.",
                )
            tools_list.append(tavily_tool)
            print("Tavily search tool initialized successfully")
        except Exception as e:
            print(f"Tavily search not available: {e}")

    # DuckDuckGo requires no key and is the always-on web search fallback
    try:
        tools_list.append(DuckDuckGoSearchRun(
            api_wrapper=DuckDuckGoSearchAPIWrapper(max_results=5),
            description="Search the web using DuckDuckGo for current information and news.",
        ))
        print("DuckDuckGo search tool initialized successfully")
    except Exception as e:
        print(f"DuckDuckGo not available: {e}")

    tools_list.extend([
        calculator,
        code_analyzer,
        weather_info,
        file_content_generator,
        semantic_scholar_search,
        openalex_search,
    ])

    print(f"Initialized {len(tools_list)} tools successfully")
    return tools_list
