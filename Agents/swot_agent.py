"""
agents/swot_agent.py
====================
Agent 3: SWOT Analysis Agent

RESPONSIBILITY:
Synthesizes outputs from Agent 1 (Market Research) + Agent 2 (Competitor)
to generate a rich SWOT analysis.

KEY INSIGHT:
This agent doesn't need web search — it reasons over previous agents' outputs.
This is "chained reasoning" — each agent builds on the last.
Pure LLM synthesis from structured context.
"""

import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from state.research_state import ResearchState, SWOTOutput
from tools.rag_retriever import RAGRetrieverTool
from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


def swot_agent(state: ResearchState) -> dict:
    """LangGraph node: SWOT Analysis Agent."""
    logger.info(f"[SWOT] Starting for: {state['startup_idea']}")

    idea = state["startup_idea"]
    market = state.get("market_research")
    competitor = state.get("competitor_analysis")

    market_summary = ""
    if market:
        market_summary = f"""
Industry: {market.industry_overview[:300]}
Market Size: {market.market_size}
Key Trends: {', '.join(market.market_trends[:3])}
Opportunities: {', '.join(market.opportunities[:2])}
"""

    competitor_summary = ""
    if competitor:
        competitor_summary = f"""
Top Competitors: {', '.join([c.get('name','?') for c in competitor.top_competitors[:3]])}
Market Gaps: {', '.join(competitor.market_gaps[:3])}
Threat Level: {competitor.threat_level}
Competitive Advantages: {', '.join(competitor.competitive_advantages[:2])}
"""

    llm = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.gemini_api_key,
        temperature=0.6,
    )

    retriever = RAGRetrieverTool()
    rag_context = retriever.retrieve_for_agent("swot", idea)

    system_prompt = """You are a strategic business consultant specializing in startup analysis.
Generate comprehensive SWOT analyses with actionable strategic recommendations.
Always respond with valid JSON only."""

    user_prompt = f"""Generate a detailed SWOT analysis for: "{idea}"

MARKET RESEARCH FINDINGS:
{market_summary}

COMPETITOR INTELLIGENCE:
{competitor_summary}

SWOT FRAMEWORK REFERENCE:
{rag_context}

Return ONLY valid JSON:
{{
  "strengths": ["strength1", "strength2", "strength3", "strength4"],
  "weaknesses": ["weakness1", "weakness2", "weakness3", "weakness4"],
  "opportunities": ["opportunity1", "opportunity2", "opportunity3", "opportunity4"],
  "threats": ["threat1", "threat2", "threat3", "threat4"],
  "strategic_recommendations": [
    "SO strategy: use strength X to capture opportunity Y",
    "ST strategy: use strength X to counter threat Y",
    "WO strategy: fix weakness X to pursue opportunity Y",
    "WT strategy: defensive move for weakness X against threat Y"
  ]
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
        output = SWOTOutput(**data)
        logger.info(f"[SWOT] Done. {len(output.strengths)} strengths, {len(output.threats)} threats")
    except Exception as e:
        logger.error(f"[SWOT] Failed: {e}")
        output = SWOTOutput(
            strengths=["Innovative approach", "Tech-first solution", "Niche focus"],
            weaknesses=["Limited brand recognition", "Bootstrapped initially"],
            opportunities=["Growing market", "Digital adoption trends"],
            threats=["Established competitors", "Market saturation risk"],
            strategic_recommendations=["Focus on niche first", "Build community early"]
        )

    return {
        "swot_analysis": output,
        "current_agent": "swot",
        "completed_agents": state.get("completed_agents", []) + ["swot"],
    }
