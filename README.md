# 📄 Resume RAG Chatbot

A multi-agent conversational AI system that provides intelligent search and analysis of candidate resumes using Retrieval-Augmented Generation (RAG), built with LangGraph, Qdrant, and OpenAI.

## 🌟 Features

- **Multi-Agent Architecture**: Intelligent routing between RAG and general chat agents
- **Semantic Resume Search**: Vector-based similarity search across resume database
- **Smart Follow-up**: Context-aware conversations with resume ID memory
- **Comparison Mode**: Side-by-side candidate comparisons with structured tables
- **Interactive UI**: Clean Streamlit interface with chat history and token tracking
- **Observability**: Integrated Langfuse tracing for LLM monitoring

## 🏗️ Architecture

### Agent System
```
User Query → Supervisor Agent
                ├─→ RAG Agent (resume queries)
                └─→ Chat Agent (general queries)
```

- **Supervisor Agent**: Routes queries based on intent (resume-related vs. general)
- **RAG Agent**: Retrieves and answers from resume database with context
- **Chat Agent**: Handles general conversation and guidance

### Technology Stack

- **LangGraph**: Multi-agent orchestration and state management
- **Qdrant**: Vector database for semantic search
- **OpenAI**: LLMs (GPT-4.1-mini) and embeddings (text-embedding-3-small)
- **LangChain**: LLM framework and tooling
- **Streamlit**: Web interface
- **Langfuse**: Observability and tracing (optional)

## 📋 Prerequisites

- Python 3.8+
- OpenAI API key
- Qdrant Cloud account (or local instance)
- Langfuse account (optional, for tracing)

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone <https://github.com/carolineliem904-max/resume-app.git>
cd resume-app
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

**Required packages:**
```txt
beautifulsoup4==4.14.3
langchain==1.1.3
langchain_core==1.2.0
langchain_openai==1.1.3
langfuse==3.10.6
langgraph==1.0.5
pandas==2.3.3
python-dotenv==1.2.1
qdrant_client==1.16.2
streamlit==1.51.0
typing_extensions==4.15.0
tiktoken
```

### 3. Environment Setup

Create a `.env` file in the project root:

```env
# OpenAI
OPENAI_API_KEY=sk-...

# Qdrant Cloud
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-qdrant-api-key

# Langfuse (optional)
LANGFUSE_PUBLIC_KEY=pk-...
LANGFUSE_SECRET_KEY=sk-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

## 📊 Data Preparation

### Step 1: Prepare Your Resume Dataset

Place your raw resume CSV file at:
```
/path/to/Resume.csv
```

Expected CSV structure:
- `ID`: Unique resume identifier
- `Category`: Job category (e.g., "Data Science", "Java Developer")
- `Resume_str`: Full resume text

### Step 2: Clean and Chunk Resumes

Update the file path in `cleaning.py`, then run:

```bash
python cleaning.py
```

This will:
- Remove HTML tags and special characters
- Normalize text formatting
- Split resumes into 300-word chunks
- Generate `cleaned_resume_chunks.csv`

### Step 3: Ingest to Qdrant

Update the CSV path in `ingest_qdrant.py`, then run:

```bash
python ingest_qdrant.py
```

This will:
- Create embeddings using OpenAI
- Upload vectors to Qdrant Cloud
- Create payload indexes for filtering
- Handle retries and batching automatically

Expected output:
```
✅ Collection 'resume_chunks' ready
✅ Payload indexes created
➡️ Total chunks: 2400, processing in 75 batches
🎉 All chunks uploaded to Qdrant
```

## 🎯 Usage

### Starting the Application

```bash
streamlit run main.py
```

The app will open at `http://localhost:8501`

### Example Queries

#### Single Resume Analysis
```
"Tell me about Resume ID 57667857"
"What are the key skills of candidate 57667857?"
"Show me the experience of Resume ID 11847784"
```

#### Comparison Mode
```
"Compare Resume ID 57667857 and 11847784"
"What's the difference between candidates 57667857 and 11847784?"
```

#### Semantic Search
```
"Find candidates with Python and machine learning experience"
"Who has 5+ years in data science?"
"Show me Java developers with cloud experience"
```

#### Follow-up Questions
```
User: "Find Python developers"
Bot: [Returns Resume IDs 123, 456, 789]
User: "Tell me more about the first one"
Bot: [Provides details on Resume ID 123]
```

## 🎨 UI Features

### Main Chat Interface
- Real-time message streaming
- Source attribution (RAG vs. Chat agent)
- Markdown formatting support
- Persistent conversation history

### Sidebar
- **Token Usage Tracking**: Monitor input/output/total tokens
- **Clear Conversation**: Reset chat and memory
- Cumulative cost estimation

## 🔧 Configuration

### Model Selection (`agents_graph.py`)
```python
SUPERVISOR_MODEL = "gpt-4.1-mini"  # Routing decisions
RAG_MODEL = "gpt-4.1-mini"         # Resume analysis
CHAT_MODEL = "gpt-4.1-mini"        # General chat
```

### RAG Parameters (`rag_tools.py`)
```python
top_k = 5              # Number of chunks to retrieve
max_len = 400          # Snippet length in characters
per_resume_k = 5       # Chunks per resume in comparison mode
```

### Memory Settings (`main.py`)
```python
MAX_HISTORY = 10       # Conversation window size
```

## 📁 Project Structure

```
resume-app/
├── main.py                      # Streamlit UI
├── agents_graph.py              # LangGraph multi-agent system
├── rag_tools.py                 # Qdrant search tool
├── cleaning.py                  # Data preprocessing
├── ingest_qdrant.py            # Vector database ingestion
├── .env                         # Environment variables
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🔍 How It Works

### 1. Query Routing
The supervisor agent analyzes user intent and routes to:
- **RAG Agent**: Resume-specific queries
- **Chat Agent**: General conversation

### 2. RAG Retrieval
Two modes of operation:

**ID Mode** (when Resume IDs detected):
```python
query = "Compare 57667857 and 11847784"
→ Fetch specific chunks for each ID
→ Format side-by-side comparison
```

**Semantic Mode** (general queries):
```python
query = "Python developers with ML experience"
→ Embed query vector
→ Search top-k similar chunks
→ Return ranked results
```

### 3. Response Generation
- RAG Agent uses strict system prompt for accuracy
- Formats single profiles or comparison tables
- Maintains resume ID memory for follow-ups
- Cites sources and avoids hallucination

## 🐛 Troubleshooting

### Common Issues

**"No relevant resume data found"**
- Verify Qdrant collection exists: Check Qdrant dashboard
- Ensure embeddings were created: Re-run `ingest_qdrant.py`
- Check API keys in `.env`

**Timeout errors during ingestion**
- Increase timeout in `ingest_qdrant.py`: `timeout=120`
- Reduce batch size: `BATCH_SIZE = 16`

**Token usage not showing**
- Langfuse handler may be disabled
- Check `LANGFUSE_*` variables in `.env`
- Usage still tracked in `token_usage_total`

**Follow-up questions not working**
- Resume IDs must be mentioned in previous response
- Use exact phrasing: "first one", "second candidate"
- Check `selected_resume_ids` in state

## 🚦 Performance Tips

1. **Optimize chunk size**: Adjust `max_words=300` in `cleaning.py`
2. **Tune retrieval**: Modify `top_k` for speed vs. accuracy
3. **Enable caching**: Qdrant caches frequent queries automatically
4. **Monitor costs**: Track token usage in sidebar
5. **Batch processing**: Increase `BATCH_SIZE` if API allows

## 📈 Future Enhancements

- [ ] Export comparison results to PDF
- [ ] Advanced filtering (years of experience, skills, location)
- [ ] Resume ranking and scoring
- [ ] Email integration for candidate outreach
- [ ] Multi-language support
- [ ] Resume parsing from PDF/DOCX uploads

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Test thoroughly
4. Submit a pull request


## 🙏 Acknowledgments

- Built with [LangGraph](https://github.com/langchain-ai/langgraph)
- Vector search powered by [Qdrant](https://qdrant.tech)
- LLM capabilities from [OpenAI](https://openai.com)
- Observability by [Langfuse](https://langfuse.com)

---
