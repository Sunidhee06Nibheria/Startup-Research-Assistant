"""
agents/market_research_agent.py
================================
Agent 1: Market Research Agent

RESPONSIBILITY:
- Searches the web for industry trends
- Estimates market size (TAM/SAM/SOM)
- Identifies customer segments
- Discovers market opportunities

HOW LANGGRAPH NODES WORK:
A "node" in LangGraph is simply a Python function that:
  1. Receives the full ResearchState
  2. Does work (LLM call, tool use, etc.)
  3. Returns a PARTIAL dict to update the state

LangGraph merges the returned dict into the shared state automatically.
"""

import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from state.research_state import ResearchState, MarketResearchOutput
from tools.web_search import WebSearchTool
from tools.rag_retriever import RAGRetrieverTool
from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


def market_research_agent(state: ResearchState) -> dict:
    """
    LangGraph node: Market Research Agent.
    
    Args:
        state: Full ResearchState from the graph
        
    Returns:
        Partial state update dict
    """
    logger.info(f"[MarketResearch] Starting for: {state['startup_idea']}")

    idea = state["startup_idea"]
    llm = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.gemini_api_key,
        temperature=settings.temperature,
    )

    # Step 1: Web Search for real market data
    searcher = WebSearchTool(max_results=settings.max_search_results)
    search_results = searcher.multi_search([
        f"{idea} market size 2024 2025",
        f"{idea} industry trends growth",
        f"{idea} target customers segments"
    ])

    # Step 2: RAG retrieval for market sizing frameworks
    retriever = RAGRetrieverTool()
    rag_context = retriever.retrieve_for_agent("market_research", idea)

    # Step 3: LLM reasoning with all gathered context
    system_prompt = """You are an expert market research analyst and startup advisor.
You analyze startup ideas and provide detailed, data-driven market research reports.
Always respond with valid JSON only — no markdown, no extra text."""

    user_prompt = f"""Analyze the startup idea: "{idea}"

WEB SEARCH FINDINGS:
{search_results}

BUSINESS FRAMEWORKS (from knowledge base):
{rag_context}

Based on the above, provide a comprehensive market research analysis.
Return ONLY valid JSON matching this exact schema:
{{
  "industry_overview": "2-3 paragraph overview of the industry",
  "market_size": "Estimated TAM, SAM, SOM with reasoning",
  "target_segments": ["segment1", "segment2", "segment3", "segment4"],
  "market_trends": ["trend1", "trend2", "trend3", "trend4", "trend5"],
  "opportunities": ["opportunity1", "opportunity2", "opportunity3"],
  "market_score": 7.5
}}"""

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        raw = response.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)
        output = MarketResearchOutput(**data)
        logger.info(f"[MarketResearch] Completed. Market score: {output.market_score}")
    except Exception as e:
        logger.error(f"[MarketResearch] Failed: {e}")
        output = MarketResearchOutput(
            industry_overview=f"Market research for {idea}. Analysis encountered an error: {str(e)[:100]}",
            market_size="Unable to estimate at this time.",
            target_segments=["General consumers", "Early adopters"],
            market_trends=["Digital transformation", "AI adoption"],
            opportunities=["First-mover advantage in niche"],
            market_score=5.0
        )

    return {
        "market_research": output,
        "current_agent": "market_research",
        "completed_agents": state.get("completed_agents", []) + ["market_research"],
        "scores": {**state.get("scores", {}), "market_score": output.market_score}
    }
