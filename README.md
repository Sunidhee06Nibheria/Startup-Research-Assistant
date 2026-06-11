# 🚀 Multi-Agent Startup Research Assistant

> **AI-powered business intelligence platform that analyzes startup ideas using 6 specialized AI agents, LangGraph orchestration, RAG knowledge retrieval, and real-time web search.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-green.svg)](https://langchain-ai.github.io/langgraph/)
[![Gemini](https://img.shields.io/badge/Google_Gemini-1.5_Flash-orange.svg)](https://aistudio.google.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38+-red.svg)](https://streamlit.io)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_DB-purple.svg)](https://github.com/facebookresearch/faiss)

---

## 🎯 What It Does

Enter a startup idea like *"AI-powered handmade craft marketplace"* and get a full business intelligence report in minutes:

| Agent | Output |
|-------|--------|
| 📊 **Market Research** | TAM/SAM/SOM, trends, customer segments |
| 🕵️ **Competitor Intel** | Real competitors, pricing, market gaps |
| ⚖️ **SWOT Analysis** | Strategic strengths, weaknesses, recommendations |
| 💰 **Revenue Model** | Pricing tiers, revenue streams, Year 1 estimate |
| 🚀 **GTM Strategy** | Launch plan, channels, acquisition roadmap |
| 📋 **Report Agent** | Executive summary, viability scores, PDF export |

---

## 🏗️ Architecture

```
User Input
    │
    ▼
LangGraph StateGraph (Directed Pipeline)
    │
    ├── Agent 1: Market Research ──► DuckDuckGo Search + FAISS RAG
    ├── Agent 2: Competitor Intel ──► DuckDuckGo Search + FAISS RAG  
    ├── Agent 3: SWOT Analysis ─────► FAISS RAG (synthesis from Agents 1+2)
    ├── Agent 4: Revenue Model ─────► DuckDuckGo Search + FAISS RAG
    ├── Agent 5: GTM Strategy ──────► DuckDuckGo Search + FAISS RAG
    └── Agent 6: Report Generator ──► LLM synthesis of all outputs
                 │
                 ▼
         FinalReport + Scores
         (Markdown / PDF export)
```

**State Communication:**
Each agent reads the shared `ResearchState` TypedDict and writes back partial updates. No agent calls another directly — pure state-based communication.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Google Gemini 1.5 Flash |
| Agent Orchestration | LangGraph StateGraph |
| LLM Framework | LangChain |
| Vector Database | FAISS (Facebook AI) |
| Embeddings | HuggingFace sentence-transformers/all-MiniLM-L6-v2 |
| Web Search | DuckDuckGo Search API |
| Web Scraping | BeautifulSoup4 + lxml |
| State Schema | Pydantic v2 |
| Frontend | Streamlit + Plotly |
| Export | FPDF2 (PDF), Markdown |
| Logging | Loguru |
| Testing | Pytest |
| Config | Pydantic Settings + python-dotenv |

---

## 📁 Project Structure

```
startup-research-assistant/
├── agents/                    # 6 specialized AI agents
│   ├── market_research_agent.py
│   ├── competitor_agent.py
│   ├── swot_agent.py
│   ├── revenue_agent.py
│   ├── gtm_agent.py
│   └── report_agent.py
├── workflows/
│   └── research_graph.py      # LangGraph StateGraph definition
├── state/
│   └── research_state.py      # Pydantic state models + TypedDict
├── rag/
│   ├── knowledge_loader.py    # Knowledge base + document chunking
│   ├── embeddings.py          # HuggingFace embedding model
│   └── vector_store.py        # FAISS index build/search
├── tools/
│   ├── web_search.py          # DuckDuckGo search wrapper
│   ├── web_scraper.py         # BeautifulSoup scraper
│   └── rag_retriever.py       # RAG query tool for agents
├── frontend/
│   └── dashboard.py           # Full Streamlit dashboard
├── utils/
│   ├── logger.py              # Loguru logging setup
│   ├── helpers.py             # Utility functions
│   └── report_exporter.py     # Markdown + PDF export
├── config/
│   └── settings.py            # Pydantic Settings config
├── tests/
│   ├── test_agents.py
│   ├── test_rag.py
│   └── test_workflow.py
├── data/
│   ├── knowledge_base/        # Business framework documents
│   └── faiss_index/           # Auto-generated vector index
├── reports/                   # Generated report outputs
├── app.py                     # Application entry point
├── requirements.txt
└── .env.example
```

---

## ⚡ Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/startup-research-assistant.git
cd startup-research-assistant

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure API Key

```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
# Get a free key at: https://aistudio.google.com
```

### 3. Run

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 🚀 Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo, set main file to `app.py`
4. Add `GEMINI_API_KEY` in the Secrets section
5. Deploy!

---

## 💡 Key AI Engineering Concepts Demonstrated

| Concept | Implementation |
|---------|----------------|
| **Multi-Agent Systems** | 6 specialized agents with distinct roles |
| **Agentic AI** | Agents autonomously search web and retrieve knowledge |
| **RAG (Retrieval-Augmented Generation)** | FAISS + embeddings for grounded responses |
| **LLM Orchestration** | LangGraph StateGraph manages execution flow |
| **Tool Calling** | Agents call web search, scraper, and RAG tools |
| **State Management** | TypedDict + Pydantic for typed shared state |
| **Workflow Graphs** | DAG pipeline with sequential node execution |
| **Production Code Patterns** | Logging, retry logic, error handling, config management |

---

## 📝 Resume Bullet Points

```
• Built a production-grade Multi-Agent AI system using LangGraph and Google Gemini that 
  automates startup market research across 6 specialized agents (market, competitor, 
  SWOT, revenue, GTM, report generation)

• Implemented a RAG pipeline using FAISS vector database and HuggingFace sentence 
  transformers to ground agent responses with business framework knowledge

• Engineered a LangGraph StateGraph workflow with Pydantic-typed shared state for 
  type-safe inter-agent communication and sequential chained reasoning

• Integrated real-time web search (DuckDuckGo) and BeautifulSoup scraping as agent 
  tools, enabling live market intelligence gathering

• Deployed an interactive Streamlit dashboard with Plotly visualizations, live agent 
  progress tracking, and Markdown/PDF report export
```

---

## 🎤 Recruiter Q&A

**Q: What is LangGraph?**
> LangGraph is a library for building stateful, multi-agent AI workflows as directed graphs. Each node is a Python function (agent), edges define execution order, and all agents communicate through a shared typed state dict.

**Q: How does RAG work here?**
> I pre-embedded startup business frameworks into a FAISS vector index using sentence-transformers. At runtime, each agent queries the index with a relevant question, retrieves the top-3 most semantically similar chunks, and injects them into the LLM prompt as expert context — this grounds the LLM's responses.

**Q: Why FAISS over Pinecone/Weaviate?**
> FAISS runs locally with zero external dependencies or costs — ideal for a portfolio project. In production at scale, I'd migrate to a managed vector DB like Pinecone for persistence, multi-tenancy, and metadata filtering.

**Q: How do agents communicate?**
> Via shared TypedDict state — not direct function calls. Each agent receives the full state, does its work, and returns a partial dict update. LangGraph merges the update into state. This decouples agents and mirrors Redux patterns from frontend engineering.

---

*Built as a portfolio project demonstrating production-grade AI engineering practices.*
