"""
agents/competitor_agent.py
===========================
Agent 2: Competitor Intelligence Agent

RESPONSIBILITY:
- Searches for real competitors
- Analyzes positioning and pricing
- Identifies market gaps
- Rates competitive intensity

KEY DESIGN: This agent reads market_research from state (output of Agent 1)
and builds on it — this is the "chain" in multi-agent chaining.
"""

import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from state.research_state import ResearchState, CompetitorAnalysisOutput
from tools.web_search import WebSearchTool
from tools.rag_retriever import RAGRetrieverTool
from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


def competitor_agent(state: ResearchState) -> dict:
    """LangGraph node: Competitor Intelligence Agent."""
    logger.info(f"[Competitor] Starting for: {state['startup_idea']}")

    idea = state["startup_idea"]
    market_research = state.get("market_research")
    market_context = market_research.industry_overview if market_research else ""

    llm = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.gemini_api_key,
        temperature=settings.temperature,
    )

    searcher = WebSearchTool(max_results=settings.max_search_results)
    search_results = searcher.multi_search([
        f"{idea} competitors startups companies",
        f"{idea} pricing comparison",
        f"best {idea} apps platforms 2024"
    ])

    retriever = RAGRetrieverTool()
    rag_context = retriever.retrieve_for_agent("competitor", idea)

    system_prompt = """You are an expert competitive intelligence analyst.
You research competitors thoroughly and identify strategic market gaps.
Always respond with valid JSON only."""

    user_prompt = f"""Perform competitor analysis for: "{idea}"

MARKET CONTEXT (from previous agent):
{market_context}

WEB SEARCH FINDINGS:
{search_results}

COMPETITIVE FRAMEWORKS:
{rag_context}

Return ONLY valid JSON:
{{
  "top_competitors": [
    {{"name": "CompanyName", "positioning": "what they do", "pricing": "pricing info", "weakness": "their gap"}},
    {{"name": "CompanyName2", "positioning": "what they do", "pricing": "pricing info", "weakness": "their gap"}},
    {{"name": "CompanyName3", "positioning": "what they do", "pricing": "pricing info", "weakness": "their gap"}}
  ],
  "market_gaps": ["gap1", "gap2", "gap3"],
  "competitive_advantages": ["advantage1", "advantage2", "advantage3"],
  "threat_level": "Medium",
  "competition_score": 6.0
}}"""

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)
        output = CompetitorAnalysisOutput(**data)
        logger.info(f"[Competitor] Done. Threat: {output.threat_level}, Score: {output.competition_score}")
    except Exception as e:
        logger.error(f"[Competitor] Failed: {e}")
        output = CompetitorAnalysisOutput(
            top_competitors=[{"name": "Existing players", "positioning": "Various", "pricing": "Unknown", "weakness": "Not tailored to niche"}],
            market_gaps=["Underserved niche segments", "Better UX needed"],
            competitive_advantages=["First-mover in niche", "Better technology"],
            threat_level="Medium",
            competition_score=5.0
        )

    return {
        "competitor_analysis": output,
        "current_agent": "competitor",
        "completed_agents": state.get("completed_agents", []) + ["competitor"],
        "scores": {**state.get("scores", {}), "competition_score": output.competition_score}
    }
