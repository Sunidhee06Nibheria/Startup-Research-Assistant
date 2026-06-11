"""
agents/gtm_agent.py
====================
Agent 5: Go-To-Market Strategy Agent

RESPONSIBILITY:
- Designs the launch strategy
- Recommends marketing channels
- Creates a phased growth plan
- Builds customer acquisition strategy
"""

import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from state.research_state import ResearchState, GTMOutput
from tools.web_search import WebSearchTool
from tools.rag_retriever import RAGRetrieverTool
from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


def gtm_agent(state: ResearchState) -> dict:
    """LangGraph node: Go-To-Market Agent."""
    logger.info(f"[GTM] Starting for: {state['startup_idea']}")

    idea = state["startup_idea"]
    market = state.get("market_research")
    revenue = state.get("revenue_model")

    context = ""
    if market:
        context += f"Target Segments: {', '.join(market.target_segments[:3])}\n"
    if revenue:
        context += f"Revenue Model: {revenue.recommended_model}\n"
        context += f"Pricing: {revenue.pricing_tiers[0] if revenue.pricing_tiers else 'TBD'}\n"

    llm = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.gemini_api_key,
        temperature=0.7,
    )

    searcher = WebSearchTool()
    search_results = searcher.search_formatted(f"go to market strategy {idea} startup launch")

    retriever = RAGRetrieverTool()
    rag_context = retriever.retrieve_for_agent("gtm", idea)

    system_prompt = """You are a growth marketing expert and startup GTM strategist.
You design practical, actionable go-to-market plans for early-stage startups.
Always respond with valid JSON only."""

    user_prompt = f"""Create a Go-To-Market strategy for: "{idea}"

BUSINESS CONTEXT:
{context}

MARKET SEARCH FINDINGS:
{search_results}

GTM FRAMEWORKS:
{rag_context}

Return ONLY valid JSON:
{{
  "launch_strategy": "Detailed launch approach (2-3 sentences)",
  "primary_channels": ["channel1", "channel2", "channel3", "channel4"],
  "growth_tactics": ["tactic1", "tactic2", "tactic3", "tactic4", "tactic5"],
  "acquisition_plan": "How to acquire first 100, then 1000 customers",
  "timeline_phases": [
    {{"phase": "Phase 1", "duration": "Month 1-3", "goals": "Beta launch, 50 users", "activities": "Closed beta, feedback loops"}},
    {{"phase": "Phase 2", "duration": "Month 4-6", "goals": "Public launch, 500 users", "activities": "Product Hunt, content marketing"}},
    {{"phase": "Phase 3", "duration": "Month 7-12", "goals": "1000+ users, $10k MRR", "activities": "Paid channels, partnerships"}}
  ],
  "gtm_readiness_score": 7.5
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
        output = GTMOutput(**data)
        logger.info(f"[GTM] Done. Channels: {len(output.primary_channels)}, Score: {output.gtm_readiness_score}")
    except Exception as e:
        logger.error(f"[GTM] Failed: {e}")
        output = GTMOutput(
            launch_strategy="Start with a closed beta to gather feedback, then public launch via Product Hunt.",
            primary_channels=["Content marketing", "Social media", "SEO", "Referrals"],
            growth_tactics=["Influencer partnerships", "Free tier viral loop", "Community building"],
            acquisition_plan="Target early adopters via Reddit/LinkedIn communities, then expand via word-of-mouth.",
            timeline_phases=[
                {"phase": "Phase 1", "duration": "Month 1-3", "goals": "50 beta users"},
                {"phase": "Phase 2", "duration": "Month 4-6", "goals": "500 users, product-market fit"},
            ],
            gtm_readiness_score=6.5
        )

    return {
        "gtm_strategy": output,
        "current_agent": "gtm",
        "completed_agents": state.get("completed_agents", []) + ["gtm"],
        "scores": {**state.get("scores", {}), "gtm_score": output.gtm_readiness_score}
    }
