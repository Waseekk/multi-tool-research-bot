# 🤖 Multi-Tool Research Bot

An intelligent AI assistant built with Streamlit, LangChain, and LangGraph that provides access to multiple research and utility tools.

![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)

## 🌟 Features

### Research Tools
- **📚 ArXiv Search**: Access to academic papers and preprints
- **📖 Wikipedia**: General knowledge and factual information
- **🌐 Web Search**: Current information via Tavily and DuckDuckGo

### Utility Tools
- **🧮 Calculator**: Safe mathematical calculations
- **💻 Code Analyzer**: Basic code syntax analysis and feedback
- **🌤️ Weather Info**: Location-based weather information
- **📄 File Generator**: Generate sample files (CSV, JSON, Python, Markdown)

### Advanced Features
- **🧠 Context Awareness**: Maintains conversation context and memory
- **🔄 Error Handling**: Robust fallback mechanisms
- **📊 Multiple LLM Models**: Automatic fallback between Groq models
- **💾 Session Management**: Persistent chat history

## 🚀 Live Demo

[View Live Demo](https://multi-tool-research-bot-5pmekpzhczcrdtichq3rw4.streamlit.app/) 


## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Groq API key (required)
- Tavily API key (optional, for enhanced web search)

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/multi-tool-research-bot.git
   cd multi-tool-research-bot
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

## 🔧 Configuration

### Required API Keys

1. **Groq API Key** (Required)
   - Sign up at [Groq Console](https://console.groq.com/)
   - Create an API key
   - Add to `.env` file: `GROQ_API_KEY=your_key_here`

2. **Tavily API Key** (Optional - for enhanced web search)
   - Sign up at [Tavily](https://tavily.com/)
   - Add to `.env` file: `TAVILY_API_KEY=your_key_here`

### Environment Variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here  # Optional
```

## 📁 Project Structure

```
multi-tool-research-bot/
├── app.py                 # Main Streamlit application
├── src/
│   ├── __init__.py       # Package initialization
│   ├── tools.py          # Tool definitions and initialization
│   ├── models.py         # LLM models and state management
│   ├── nodes.py          # LangGraph nodes
│   └── conversation.py   # Conversation management
├── requirements.txt      # Python dependencies
├── .env.example         # Environment variables template
├── .gitignore           # Git ignore file
└── README.md            # Project documentation
```

## 🎯 Usage Examples

### Basic Calculations
```
User: Calculate 15% of 2,500
Bot: Using the calculator tool... Result: 375.0
```

### Research Queries
```
User: Latest research on quantum computing
Bot: Searching ArXiv for recent papers... 
[Returns latest quantum computing research papers with abstracts]
```

### Code Analysis
```
User: Analyze this Python code: def hello(): print('Hi')
Bot: Code Analysis for python:
- Lines of code: 1
- Contains functions: True
- Syntax: Valid Python syntax
```

### File Generation
```
User: Generate a CSV file for student data
Bot: # Sample CSV for: student data
name,age,city,score
Alice,25,New York,85
Bob,30,London,92
...
```

## 🏗️ Architecture

The application uses a modern architecture with:

- **Streamlit**: Web interface and user interaction
- **LangChain**: Tool integration and LLM orchestration
- **LangGraph**: Conversation flow and state management
- **Groq**: Fast LLM inference with multiple model fallbacks

### Key Components

1. **Tools Module** (`src/tools.py`)
   - Custom tools for calculations, code analysis, weather, file generation
   - Integration with ArXiv, Wikipedia, and web search APIs

2. **Models Module** (`src/models.py`)
   - LLM management with automatic fallbacks
   - Conversation state schema

3. **Nodes Module** (`src/nodes.py`)
   - LangGraph nodes for conversation flow
   - Error handling and context management

4. **Conversation Module** (`src/conversation.py`)
   - Chat history management
   - Context summarization

## 🚀 Deployment

### Deploy to Streamlit Cloud

1. Fork this repository
2. Connect your GitHub account to [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy from your forked repository
4. Add your API keys in the Streamlit Cloud secrets

### Deploy to Other Platforms

- **Heroku**: Add `runtime.txt` with Python version
- **Docker**: Use the provided Dockerfile
- **Railway**: Connect GitHub repository directly

### Docker Deployment

```bash
# Build image
docker build -t multi-tool-research-bot .

# Run container
docker run -p 8501:8501 --env-file .env multi-tool-research-bot
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/

# Run linting
flake8 src/
black src/
```

## 📊 Features Roadmap

- [ ] **Enhanced Search**: Add more search engines and APIs
- [ ] **Real Weather API**: Integrate with OpenWeatherMap
- [ ] **Document Upload**: Support PDF/DOCX file analysis
- [ ] **Data Visualization**: Generate charts and graphs
- [ ] **Export Features**: Save conversations and results
- [ ] **Voice Interface**: Speech-to-text integration
- [ ] **Multi-language**: Support for multiple languages

## 🐛 Known Issues

- DuckDuckGo search may occasionally rate limit
- Weather information is currently simulated
- Code analysis is basic (syntax checking only)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) for the amazing web framework
- [LangChain](https://langchain.com/) for LLM orchestration
- [Groq](https://groq.com/) for fast LLM inference
- [ArXiv](https://arxiv.org/) for academic paper access
- [Wikipedia](https://www.wikipedia.org/) for knowledge base

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/multi-tool-research-bot/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/multi-tool-research-bot/discussions)
- **Email**: your.email@example.com

---

**Built with ❤️ for the AI community**