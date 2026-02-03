import os
import json
import random
from datetime import datetime
from typing import List

# Environment setup
from dotenv import load_dotenv
load_dotenv()

from langchain_core.tools import tool
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, PubmedQueryRun
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper, PubMedAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

# Try to import updated Tavily
try:
    from langchain_tavily import TavilySearch
    TAVILY_AVAILABLE = True
except ImportError:
    try:
        # Fallback to old import if new one not available
        from langchain_community.tools.tavily_search import TavilySearchResults
        TAVILY_AVAILABLE = True
    except ImportError:
        TAVILY_AVAILABLE = False
        print("Tavily search not available. Install with: pip install langchain-tavily")

@tool
def calculator(expression: str) -> str:
    """
    Perform mathematical calculations safely.
    
    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 3 * 4", "15/100 * 2500")
    
    Returns:
        Result of the calculation or error message
    """
    try:
        # Safe evaluation - only allow basic math operations
        allowed_chars = set('0123456789+-*/.() ')
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression. Only numbers and basic operators (+, -, *, /, parentheses) are allowed."
        
        # Replace common percentage patterns
        if '%' in expression:
            return "Error: Use decimal form instead of %. For example: '15/100 * 2500' instead of '15% * 2500'"
        
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error calculating '{expression}': {str(e)}"

@tool
def code_analyzer(code: str, language: str = "python") -> str:
    """
    Analyze code for basic syntax and provide simple feedback.
    
    Args:
        code: Code snippet to analyze
        language: Programming language (default: python)
    
    Returns:
        Basic analysis of the code
    """
    try:
        analysis = {
            "language": language,
            "lines": len(code.split('\n')),
            "characters": len(code),
            "contains_functions": "def " in code if language == "python" else "function " in code,
            "contains_classes": "class " in code if language == "python" else False,
            "contains_imports": any(line.strip().startswith(('import ', 'from ')) for line in code.split('\n')) if language == "python" else False
        }
        
        feedback = f"""
Code Analysis for {language}:
- Lines of code: {analysis['lines']}
- Total characters: {analysis['characters']}
- Contains functions: {analysis['contains_functions']}
- Contains classes: {analysis['contains_classes']}
- Contains imports: {analysis['contains_imports']}
        """
        
        if language == "python":
            try:
                compile(code, '<string>', 'exec')
                feedback += "\n- Syntax: Valid Python syntax"
            except SyntaxError as e:
                feedback += f"\n- Syntax Error: {str(e)}"
        
        return feedback.strip()
    except Exception as e:
        return f"Error analyzing code: {str(e)}"

@tool
def weather_info(location: str) -> str:
    """
    Get weather information for a location (simulated for demo).
    Note: This is a demo function. In production, integrate with a real weather API.
    
    Args:
        location: City or location name
    
    Returns:
        Weather information
    """
    # Simulated weather data - in real implementation, use weather API
    weather_conditions = ["sunny", "cloudy", "rainy", "partly cloudy", "windy"]
    temperature = random.randint(15, 35)
    condition = random.choice(weather_conditions)
    
    return f"""
Weather for {location} (Demo Data):
- Temperature: {temperature}°C
- Condition: {condition.title()}
- Humidity: {random.randint(40, 80)}%
- Wind Speed: {random.randint(5, 25)} km/h
Note: This is simulated data for demonstration purposes.
    """.strip()

@tool
def semantic_scholar_search(query: str, limit: int = 5) -> str:
    """
    Search Semantic Scholar for academic papers across all scientific fields.
    Great for finding highly-cited papers and understanding research impact.

    Args:
        query: Search query for academic papers
        limit: Number of results to return (default: 5)

    Returns:
        Formatted list of papers with title, authors, year, and citation count
    """
    import requests

    try:
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": min(limit, 10),
            "fields": "title,abstract,authors,year,citationCount,url"
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if not data.get("data"):
            return f"No papers found for query: {query}"

        results = []
        for i, paper in enumerate(data["data"], 1):
            title = paper.get("title", "Unknown Title")
            year = paper.get("year", "N/A")
            citations = paper.get("citationCount", 0)
            url = paper.get("url", "")

            authors = paper.get("authors", [])
            author_names = ", ".join([a.get("name", "") for a in authors[:3]])
            if len(authors) > 3:
                author_names += " et al."

            abstract = paper.get("abstract", "")
            if abstract and len(abstract) > 300:
                abstract = abstract[:300] + "..."

            result = f"""
**{i}. {title}**
- Authors: {author_names}
- Year: {year} | Citations: {citations}
- URL: {url}
- Abstract: {abstract if abstract else 'No abstract available'}
"""
            results.append(result)

        return f"## Semantic Scholar Results for '{query}':\n" + "\n".join(results)

    except requests.exceptions.Timeout:
        return "Error: Semantic Scholar API request timed out. Please try again."
    except requests.exceptions.RequestException as e:
        return f"Error searching Semantic Scholar: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def openalex_search(query: str, limit: int = 5) -> str:
    """
    Search OpenAlex for scholarly works across all academic disciplines.
    OpenAlex is a free, open catalog of the world's scholarly papers, authors, and institutions.

    Args:
        query: Search query for academic works
        limit: Number of results to return (default: 5)

    Returns:
        Formatted list of works with title, authors, year, and open access status
    """
    import requests

    try:
        url = "https://api.openalex.org/works"
        params = {
            "search": query,
            "per-page": min(limit, 10),
            "select": "title,authorships,publication_year,cited_by_count,open_access,doi"
        }
        headers = {
            "User-Agent": "MultiToolResearchBot/1.0 (mailto:research@example.com)"
        }

        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        if not data.get("results"):
            return f"No works found for query: {query}"

        results = []
        for i, work in enumerate(data["results"], 1):
            title = work.get("title", "Unknown Title")
            year = work.get("publication_year", "N/A")
            citations = work.get("cited_by_count", 0)
            doi = work.get("doi", "")

            # Get authors
            authorships = work.get("authorships", [])
            author_names = ", ".join([
                a.get("author", {}).get("display_name", "Unknown")
                for a in authorships[:3]
            ])
            if len(authorships) > 3:
                author_names += " et al."

            # Open access status
            oa = work.get("open_access", {})
            oa_status = "Open Access" if oa.get("is_oa") else "Closed Access"
            oa_url = oa.get("oa_url", "")

            result = f"""
**{i}. {title}**
- Authors: {author_names}
- Year: {year} | Citations: {citations}
- Access: {oa_status}
- DOI: {doi if doi else 'N/A'}
{f'- Open Access URL: {oa_url}' if oa_url else ''}
"""
            results.append(result)

        return f"## OpenAlex Results for '{query}':\n" + "\n".join(results)

    except requests.exceptions.Timeout:
        return "Error: OpenAlex API request timed out. Please try again."
    except requests.exceptions.RequestException as e:
        return f"Error searching OpenAlex: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def file_content_generator(file_type: str, content_description: str) -> str:
    """
    Generate sample file content based on type and description.
    
    Args:
        file_type: Type of file (e.g., 'csv', 'json', 'python', 'markdown')
        content_description: Description of what the file should contain
    
    Returns:
        Generated file content
    """
    try:
        if file_type.lower() == 'csv':
            return f"""# Sample CSV for: {content_description}
name,age,city,score
Alice,25,New York,85
Bob,30,London,92
Charlie,35,Tokyo,78
Diana,28,Paris,88"""

        elif file_type.lower() == 'json':
            sample_data = {
                "description": content_description,
                "data": [
                    {"id": 1, "name": "Item 1", "value": 100},
                    {"id": 2, "name": "Item 2", "value": 250},
                    {"id": 3, "name": "Item 3", "value": 175}
                ],
                "metadata": {
                    "created": datetime.now().isoformat(),
                    "version": "1.0"
                }
            }
            return json.dumps(sample_data, indent=2)

        elif file_type.lower() == 'python':
            return f'''"""
{content_description}
"""

def main():
    """Main function for {content_description}"""
    print("Hello, World!")
    
    # Add your code here
    data = [1, 2, 3, 4, 5]
    result = process_data(data)
    print(f"Result: {{result}}")

def process_data(data):
    """Process the input data"""
    return sum(data)

if __name__ == "__main__":
    main()
'''

        elif file_type.lower() == 'markdown':
            return f"""# {content_description}

## Overview
This document covers {content_description.lower()}.

## Key Points
- Point 1: Important information
- Point 2: Additional details
- Point 3: Summary notes

## Code Example
```python
def example_function():
    return "Hello, World!"
```

## Conclusion
This covers the basics of {content_description.lower()}.
"""

        else:
            return f"Sample content for {file_type} file:\n{content_description}\n\nGenerated on: {datetime.now()}"
            
    except Exception as e:
        return f"Error generating {file_type} content: {str(e)}"

def initialize_tools() -> List:
    """Initialize all available tools"""
    
    # Academic research tools
    arxiv_wrapper = ArxivAPIWrapper(top_k_results=10, doc_content_chars_max=10000)
    arxiv_tool = ArxivQueryRun(
        api_wrapper=arxiv_wrapper,
        description="Search academic papers on arXiv. Best for recent research papers and preprints."
    )
    
    # Wikipedia for general knowledge
    wiki_wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=800)
    wiki_tool = WikipediaQueryRun(
        api_wrapper=wiki_wrapper,
        description="Search Wikipedia for general knowledge, definitions, and factual information."
    )

    # PubMed for medical/biomedical research
    pubmed_wrapper = PubMedAPIWrapper(top_k_results=5, doc_content_chars_max=1500)
    pubmed_tool = PubmedQueryRun(
        api_wrapper=pubmed_wrapper,
        description="Search PubMed for medical, biomedical, life sciences, and healthcare research papers."
    )

    # Web search tools
    tools_list = [arxiv_tool, wiki_tool, pubmed_tool]
    
    # Add Tavily if API key is available and package is installed
    if TAVILY_AVAILABLE and os.getenv("TAVILY_API_KEY"):
        try:
            # Try new import first
            try:
                tavily_tool = TavilySearch(
                    max_results=5,
                    description="Search the web for recent news, current events, and real-time information."
                )
            except NameError:
                # Fallback to old import
                tavily_tool = TavilySearchResults(
                    max_results=5,
                    description="Search the web for recent news, current events, and real-time information."
                )
            tools_list.append(tavily_tool)
            print("Tavily search tool initialized successfully")
        except Exception as e:
            print(f"Tavily search not available: {e}")
    
    # Add DuckDuckGo as fallback web search
    try:
        ddg_wrapper = DuckDuckGoSearchAPIWrapper(max_results=5)
        ddg_tool = DuckDuckGoSearchRun(
            api_wrapper=ddg_wrapper,
            description="Search the web using DuckDuckGo for current information and news."
        )
        tools_list.append(ddg_tool)
        print("DuckDuckGo search tool initialized successfully")
    except Exception as e:
        print(f"Note: DuckDuckGo search not available: {e}")
    
    # Add custom tools
    tools_list.extend([
        calculator,
        code_analyzer,
        weather_info,
        file_content_generator,
        semantic_scholar_search,
        openalex_search
    ])
    
    print(f"Initialized {len(tools_list)} tools successfully")
    return tools_list