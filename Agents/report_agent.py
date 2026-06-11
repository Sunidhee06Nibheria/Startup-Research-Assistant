"""
agents/report_agent.py
=======================
Agent 6: Report Generation Agent — the FINAL node in the graph.

RESPONSIBILITY:
- Reads ALL previous agents' outputs from state
- Writes a polished executive summary
- Calculates final composite scores
- Produces the FinalReport object

This is the "reducer" node — it aggregates everything into a deliverable.
"""

import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from state.research_state import ResearchState, FinalReport
from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


def _format_swot(swot) -> str:
    if not swot:
        return "SWOT analysis not available."
    lines = []
    lines.append("**STRENGTHS:**\n" + "\n".join(f"• {s}" for s in swot.strengths))
    lines.append("**WEAKNESSES:**\n" + "\n".join(f"• {w}" for w in swot.weaknesses))
    lines.append("**OPPORTUNITIES:**\n" + "\n".join(f"• {o}" for o in swot.opportunities))
    lines.append("**THREATS:**\n" + "\n".join(f"• {t}" for t in swot.threats))
    lines.append("**STRATEGIC RECOMMENDATIONS:**\n" + "\n".join(f"• {r}" for r in swot.strategic_recommendations))
    return "\n\n".join(lines)


def _format_competitors(competitor) -> str:
    if not competitor:
        return "Competitor analysis not available."
    lines = [f"**Threat Level:** {competitor.threat_level}\n"]
    lines.append("**Top Competitors:**")
    for c in competitor.top_competitors:
        lines.append(f"• {c.get('name','?')}: {c.get('positioning','?')} | Pricing: {c.get('pricing','?')}")
    lines.append("\n**Market Gaps:**\n" + "\n".join(f"• {g}" for g in competitor.market_gaps))
    lines.append("\n**Our Competitive Advantages:**\n" + "\n".join(f"• {a}" for a in competitor.competitive_advantages))
    return "\n".join(lines)


def _format_revenue(revenue) -> str:
    if not revenue:
        return "Revenue model not available."
    lines = [
        f"**Recommended Model:** {revenue.recommended_model}",
        f"**Pricing Strategy:** {revenue.pricing_strategy}",
        "\n**Revenue Streams:**\n" + "\n".join(f"• {s}" for s in revenue.revenue_streams),
        "\n**Pricing Tiers:**"
    ]
    for tier in revenue.pricing_tiers:
        lines.append(f"• {tier.get('name','?')}: {tier.get('price','?')} — {tier.get('features','')}")
    lines.append(f"\n**Year 1 Revenue Estimate:** {revenue.year1_revenue_estimate}")
    return "\n".join(lines)


def _format_gtm(gtm) -> str:
    if not gtm:
        return "GTM strategy not available."
    lines = [
        f"**Launch Strategy:** {gtm.launch_strategy}",
        "\n**Primary Channels:**\n" + "\n".join(f"• {c}" for c in gtm.primary_channels),
        "\n**Growth Tactics:**\n" + "\n".join(f"• {t}" for t in gtm.growth_tactics),
        f"\n**Acquisition Plan:** {gtm.acquisition_plan}",
        "\n**Timeline:**"
    ]
    for phase in gtm.timeline_phases:
        lines.append(f"• {phase.get('phase','?')} ({phase.get('duration','?')}): {phase.get('goals','?')}")
    return "\n".join(lines)


def report_agent(state: ResearchState) -> dict:
    """LangGraph node: Report Generation Agent (terminal node)."""
    logger.info(f"[Report] Generating final report for: {state['startup_idea']}")

    idea = state["startup_idea"]
    market = state.get("market_research")
    competitor = state.get("competitor_analysis")
    swot = state.get("swot_analysis")
    revenue = state.get("revenue_model")
    gtm = state.get("gtm_strategy")
    scores = state.get("scores", {})

    # Calculate composite scores
    score_values = [v for v in scores.values() if isinstance(v, (int, float))]
    overall_score = round(sum(score_values) / len(score_values), 1) if score_values else 5.0
    investment_score = round(
        (scores.get("market_score", 5) * 0.3) +
        (scores.get("viability_score", 5) * 0.3) +
        (scores.get("gtm_score", 5) * 0.2) +
        ((10 - scores.get("competition_score", 5)) * 0.2),
        1
    )

    # Build structured sections
    market_section = ""
    if market:
        market_section = f"""{market.industry_overview}

**Market Size:** {market.market_size}

**Target Segments:** {', '.join(market.target_segments)}

**Key Trends:**
{chr(10).join(f'• {t}' for t in market.market_trends)}

**Market Opportunities:**
{chr(10).join(f'• {o}' for o in market.opportunities)}"""

    # Generate executive summary using LLM
    llm = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.gemini_api_key,
        temperature=0.5,
    )

    exec_summary_prompt = f"""Write a concise, compelling executive summary (3-4 paragraphs) for a startup idea.

Startup Idea: {idea}
Overall Viability Score: {overall_score}/10
Investment Readiness Score: {investment_score}/10
Market Score: {scores.get('market_score', 'N/A')}/10
Competition Level: {competitor.threat_level if competitor else 'Unknown'}
Revenue Model: {revenue.recommended_model if revenue else 'TBD'}

Write as a professional business analyst. Be direct, insightful, and data-aware.
Mention the scores and what they mean. End with a clear recommendation."""

    try:
        exec_resp = llm.invoke([HumanMessage(content=exec_summary_prompt)])
        executive_summary = exec_resp.content
    except Exception as e:
        logger.warning(f"Executive summary LLM call failed: {e}")
        executive_summary = f"""The startup idea "{idea}" has been analyzed across market, competitive, revenue, and GTM dimensions.

Overall Viability Score: {overall_score}/10 | Investment Readiness: {investment_score}/10

Based on comprehensive research, this idea shows {'strong' if overall_score >= 7 else 'moderate' if overall_score >= 5 else 'limited'} potential in its target market. The analysis suggests focusing on early customer validation and niche market penetration before scaling.

**Recommendation:** {'Proceed with focused execution.' if overall_score >= 6 else 'Validate core assumptions before heavy investment.'}"""

    final_recommendations = ""
    if swot and swot.strategic_recommendations:
        final_recommendations = "\n".join(f"• {r}" for r in swot.strategic_recommendations)
    else:
        final_recommendations = "• Focus on product-market fit in a specific niche\n• Build early community before scaling\n• Prioritize revenue validation in first 6 months"

    report = FinalReport(
        startup_idea=idea,
        executive_summary=executive_summary,
        market_research=market_section or "Market research data not available.",
        competitor_analysis=_format_competitors(competitor),
        swot_analysis=_format_swot(swot),
        revenue_strategy=_format_revenue(revenue),
        gtm_plan=_format_gtm(gtm),
        final_recommendations=final_recommendations,
        overall_viability_score=overall_score,
        investment_readiness_score=investment_score,
    )

    logger.info(f"[Report] Final report generated. Overall: {overall_score}/10, Investment: {investment_score}/10")

    return {
        "final_report": report,
        "current_agent": "report",
        "completed_agents": state.get("completed_agents", []) + ["report"],
        "scores": {
            **scores,
            "overall_viability_score": overall_score,
            "investment_readiness_score": investment_score
        }
    }
