"""
agents/revenue_agent.py
========================
Agent 4: Revenue Model Agent

RESPONSIBILITY:
- Recommends the best revenue model for the startup
- Designs pricing tiers
- Suggests monetization strategies
- Estimates Year 1 revenue potential
"""

import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from state.research_state import ResearchState, RevenueModelOutput
from tools.web_search import WebSearchTool
from tools.rag_retriever import RAGRetrieverTool
from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


def revenue_agent(state: ResearchState) -> dict:
    """LangGraph node: Revenue Model Agent."""
    logger.info(f"[Revenue] Starting for: {state['startup_idea']}")

    idea = state["startup_idea"]
    market = state.get("market_research")
    competitor = state.get("competitor_analysis")

    context_summary = ""
    if market:
        context_summary += f"Market Size: {market.market_size}\n"
        context_summary += f"Customer Segments: {', '.join(market.target_segments[:3])}\n"
    if competitor:
        context_summary += f"Competitor Pricing: {', '.join([c.get('pricing','?') for c in competitor.top_competitors[:2]])}\n"

    llm = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.gemini_api_key,
        temperature=0.5,
    )

    searcher = WebSearchTool()
    search_results = searcher.search_formatted(f"{idea} pricing model revenue strategy")

    retriever = RAGRetrieverTool()
    rag_context = retriever.retrieve_for_agent("revenue", idea)

    system_prompt = """You are a startup monetization and revenue strategy expert.
Design practical, realistic revenue models for early-stage startups.
Always respond with valid JSON only."""

    user_prompt = f"""Design a revenue model for: "{idea}"

BUSINESS CONTEXT:
{context_summary}

MARKET RESEARCH:
{search_results}

REVENUE MODEL FRAMEWORKS:
{rag_context}

Return ONLY valid JSON:
{{
  "recommended_model": "SaaS / Marketplace / Freemium / Transactional / etc.",
  "revenue_streams": ["stream1", "stream2", "stream3"],
  "pricing_strategy": "Explanation of pricing approach",
  "pricing_tiers": [
    {{"name": "Free", "price": "$0/month", "features": "Basic features"}},
    {{"name": "Pro", "price": "$29/month", "features": "Advanced features"}},
    {{"name": "Business", "price": "$99/month", "features": "All features + support"}}
  ],
  "year1_revenue_estimate": "$X - $Y ARR based on assumptions",
  "viability_score": 7.0
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
        output = RevenueModelOutput(**data)
        logger.info(f"[Revenue] Done. Model: {output.recommended_model}, Score: {output.viability_score}")
    except Exception as e:
        logger.error(f"[Revenue] Failed: {e}")
        output = RevenueModelOutput(
            recommended_model="Freemium SaaS",
            revenue_streams=["Subscription fees", "Premium features", "Enterprise plans"],
            pricing_strategy="Freemium entry, convert to paid via value demonstration",
            pricing_tiers=[
                {"name": "Free", "price": "$0", "features": "Basic access"},
                {"name": "Pro", "price": "$29/mo", "features": "Full access"},
            ],
            year1_revenue_estimate="$50,000 - $200,000 ARR",
            viability_score=6.0
        )

    return {
        "revenue_model": output,
        "current_agent": "revenue",
        "completed_agents": state.get("completed_agents", []) + ["revenue"],
        "scores": {**state.get("scores", {}), "viability_score": output.viability_score}
    }
