import os
import json
import random
from datetime import datetime
from typing import List

# Environment setup
from dotenv import load_dotenv
load_dotenv()

from langchain_core.tools import tool
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
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
    
    # Web search tools
    tools_list = [arxiv_tool, wiki_tool]
    
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
        file_content_generator
    ])
    
    print(f"Initialized {len(tools_list)} tools successfully")
    return tools_list