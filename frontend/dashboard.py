"""
frontend/dashboard.py
======================
Main Streamlit dashboard — the full UI for the application.

PAGES:
1. Home — Hero page with product overview
2. Analyze — Input + live agent progress + results
3. Report — Full formatted report with download
4. About — Architecture explanation

STREAMLIT CONCEPTS USED:
- st.session_state: Persists data across reruns (like React state)
- st.spinner / st.status: Live progress indicators
- st.tabs: Multi-section layout
- st.plotly_chart: Interactive score visualizations
- st.download_button: Export report files
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import time
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workflows.research_graph import run_research_pipeline_streaming
from utils.report_exporter import export_markdown, export_pdf
from utils.helpers import score_to_emoji, score_to_label
from utils.logger import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="StartupAI — Multi-Agent Research Assistant",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; }

    /* Card style */
    .metric-card {
        background: linear-gradient(135deg, #1a1d2e, #16213e);
        border: 1px solid #2d3561;
        border-radius: 12px;
        padding: 20px;
        margin: 8px 0;
    }

    /* Score badge */
    .score-badge {
        display: inline-block;
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white;
        font-size: 2rem;
        font-weight: bold;
        padding: 8px 20px;
        border-radius: 8px;
    }

    /* Agent status cards */
    .agent-active {
        border-left: 4px solid #667eea;
        background: #1a1d2e;
        padding: 12px;
        border-radius: 8px;
        margin: 4px 0;
    }
    .agent-done {
        border-left: 4px solid #48bb78;
        background: #1a2e1a;
        padding: 12px;
        border-radius: 8px;
        margin: 4px 0;
    }
    .agent-pending {
        border-left: 4px solid #4a5568;
        background: #1a1d2e;
        padding: 12px;
        border-radius: 8px;
        margin: 4px 0;
        opacity: 0.6;
    }

    /* Section headers */
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #667eea;
        border-bottom: 2px solid #2d3561;
        padding-bottom: 8px;
        margin: 20px 0 12px 0;
    }

    /* Hero gradient text */
    .hero-title {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 900;
        line-height: 1.1;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Session State Initialization
# ─────────────────────────────────────────────
def init_session_state():
    defaults = {
        "page": "home",
        "research_complete": False,
        "final_state": None,
        "running": False,
        "completed_agents": [],
        "current_agent": None,
        "startup_idea": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─────────────────────────────────────────────
# Sidebar Navigation
# ─────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("## 🚀 StartupAI")
        st.markdown("*Multi-Agent Research Assistant*")
        st.divider()

        pages = {
            "🏠 Home": "home",
            "🔬 Analyze Idea": "analyze",
            "📊 View Report": "report",
            "🏗️ Architecture": "architecture"
        }
        for label, page_id in pages.items():
            if st.button(label, use_container_width=True, key=f"nav_{page_id}"):
                st.session_state.page = page_id
                st.rerun()

        st.divider()

        if st.session_state.research_complete and st.session_state.final_state:
            report = st.session_state.final_state.get("final_report")
            if report:
                scores = st.session_state.final_state.get("scores", {})
                st.markdown("### 📈 Quick Scores")
                st.metric("Overall Viability", f"{report.overall_viability_score}/10")
                st.metric("Investment Ready", f"{report.investment_readiness_score}/10")

        st.divider()
        st.markdown("""
        <small>
        Built with:<br>
        🤖 Google Gemini<br>
        🔗 LangGraph<br>
        🗄️ FAISS RAG<br>
        🌐 Web Search<br>
        </small>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE: Home
# ─────────────────────────────────────────────
def render_home():
    st.markdown('<div class="hero-title">Multi-Agent Startup<br>Research Assistant</div>', unsafe_allow_html=True)
    st.markdown("### AI-powered business intelligence for startup founders")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **🤖 6 Specialized AI Agents**

        Market Research → Competitor Intel → SWOT → Revenue → GTM → Report
        """)
    with col2:
        st.markdown("""
        **🧠 RAG-Powered Analysis**

        FAISS vector store + HuggingFace embeddings + startup knowledge base
        """)
    with col3:
        st.markdown("""
        **🌐 Real-Time Web Search**

        DuckDuckGo search + web scraping for live market data
        """)

    st.markdown("---")

    st.markdown("### ⚡ How It Works")
    steps = [
        ("1️⃣ Enter Your Idea", "Describe your startup concept in a sentence"),
        ("2️⃣ Agents Activate", "6 AI agents research in sequence, each building on the last"),
        ("3️⃣ RAG Retrieval", "Each agent pulls relevant frameworks from the knowledge base"),
        ("4️⃣ Web Intelligence", "Agents search the web for real market data and competitors"),
        ("5️⃣ Report Generated", "A professional business intelligence report is compiled"),
        ("6️⃣ Export & Score", "Download as Markdown/PDF with viability scores"),
    ]
    cols = st.columns(3)
    for i, (title, desc) in enumerate(steps):
        with cols[i % 3]:
            st.info(f"**{title}**\n\n{desc}")

    st.markdown("---")
    if st.button("🚀 Start Analyzing Your Idea", use_container_width=True, type="primary"):
        st.session_state.page = "analyze"
        st.rerun()


# ─────────────────────────────────────────────
# PAGE: Analyze
# ─────────────────────────────────────────────
AGENT_CONFIG = [
    ("market_research",    "📊 Market Research Agent",     "Analyzing industry trends, market size, customer segments"),
    ("competitor_analysis","🕵️ Competitor Intelligence",   "Identifying competitors, pricing, market gaps"),
    ("swot_analysis",      "⚖️ SWOT Analysis Agent",        "Generating strengths, weaknesses, opportunities, threats"),
    ("revenue_model",      "💰 Revenue Model Agent",        "Designing pricing strategy and revenue streams"),
    ("gtm_strategy",       "🚀 Go-To-Market Agent",         "Building launch and growth strategy"),
    ("report_generation",  "📋 Report Generation Agent",    "Compiling final business intelligence report"),
]


def render_agent_status(completed: list, current: str):
    """Render the live agent pipeline status."""
    st.markdown("### 🤖 Agent Pipeline")
    for agent_id, agent_name, agent_desc in AGENT_CONFIG:
        if agent_id in completed:
            st.markdown(f'<div class="agent-done">✅ {agent_name}<br><small>{agent_desc}</small></div>', unsafe_allow_html=True)
        elif agent_id == current:
            st.markdown(f'<div class="agent-active">⚡ {agent_name} <em>(running...)</em><br><small>{agent_desc}</small></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="agent-pending">⏳ {agent_name}<br><small>{agent_desc}</small></div>', unsafe_allow_html=True)


def render_analyze():
    st.title("🔬 Analyze Your Startup Idea")
    st.markdown("Enter your startup concept and let 6 AI agents research it for you.")
    st.markdown("---")

    idea = st.text_input(
        "💡 Your Startup Idea",
        placeholder="e.g. AI-powered handmade craft marketplace for college students",
        key="idea_input",
        value=st.session_state.get("startup_idea", "")
    )

    context = st.text_area(
        "📝 Additional Context (optional)",
        placeholder="Target geography, existing traction, team background, budget...",
        height=80
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        run_btn = st.button("🚀 Run Full Research Pipeline", type="primary", use_container_width=True,
                            disabled=st.session_state.running)
    with col2:
        if st.session_state.research_complete:
            if st.button("📊 View Report", use_container_width=True):
                st.session_state.page = "report"
                st.rerun()

    if run_btn and idea.strip():
        st.session_state.startup_idea = idea.strip()
        st.session_state.running = True
        st.session_state.research_complete = False
        st.session_state.final_state = None
        st.session_state.completed_agents = []

        # ── Run the pipeline with streaming ──
        st.markdown("---")
        progress_col, status_col = st.columns([1, 2])

        with progress_col:
            agent_placeholder = st.empty()

        with status_col:
            log_placeholder = st.empty()
            progress_bar = st.progress(0)
            status_text = st.empty()

        logs = []
        final_state = None
        total_agents = len(AGENT_CONFIG)
        agent_order = [a[0] for a in AGENT_CONFIG]
        completed = []

        try:
            for node_name, state_update in run_research_pipeline_streaming(idea.strip(), context):
                completed.append(node_name)
                st.session_state.completed_agents = completed
                progress = len(completed) / total_agents
                progress_bar.progress(progress)

                agent_label = next((a[1] for a in AGENT_CONFIG if a[0] == node_name), node_name)
                status_text.markdown(f"**Completed:** {agent_label}")
                logs.append(f"✅ {agent_label}")
                log_placeholder.markdown("\n\n".join(logs))

                with agent_placeholder.container():
                    render_agent_status(completed, "")

                # Capture final state
                final_state = state_update

        except Exception as e:
            st.error(f"Pipeline error: {str(e)}")
            logger.error(f"Pipeline failed: {e}")
            st.session_state.running = False
            return

        # We need the full final state — re-run non-streaming to get complete state
        if final_state and "final_report" not in final_state:
            from workflows.research_graph import run_research_pipeline
            try:
                status_text.markdown("**Assembling final report...**")
                final_state = run_research_pipeline(idea.strip(), context)
            except Exception as e:
                st.error(f"Report assembly error: {str(e)}")

        st.session_state.final_state = final_state
        st.session_state.running = False
        st.session_state.research_complete = True

        progress_bar.progress(1.0)
        status_text.markdown("✅ **Research Complete!**")
        st.success("🎉 All agents finished! View your report below.")

        time.sleep(0.5)
        st.session_state.page = "report"
        st.rerun()

    elif run_btn and not idea.strip():
        st.warning("Please enter a startup idea first.")

    # Show previous results hint
    if st.session_state.research_complete and not st.session_state.running:
        st.info(f"✅ Previous analysis complete for: **{st.session_state.startup_idea}** — [View Report →](#)")


# ─────────────────────────────────────────────
# PAGE: Report
# ─────────────────────────────────────────────
def render_scores_chart(scores: dict, report):
    """Render a radar chart of all scores."""
    categories = ["Market\nOpportunity", "Revenue\nViability", "GTM\nReadiness", "Overall\nViability", "Investment\nReadiness"]
    values = [
        scores.get("market_score", 5),
        scores.get("viability_score", 5),
        scores.get("gtm_score", 5),
        report.overall_viability_score,
        report.investment_readiness_score,
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor="rgba(102, 126, 234, 0.2)",
        line=dict(color="#667eea", width=2),
        name="Scores"
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 10], tickfont=dict(color="white")),
            angularaxis=dict(tickfont=dict(color="white")),
            bgcolor="rgba(0,0,0,0)"
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        showlegend=False,
        height=350,
        margin=dict(l=60, r=60, t=40, b=40)
    )
    return fig


def render_scores_bar(scores: dict, report):
    """Render a horizontal bar chart of scores."""
    labels = ["Market Score", "Revenue Score", "GTM Score", "Overall Viability", "Investment Ready"]
    vals = [
        scores.get("market_score", 5),
        scores.get("viability_score", 5),
        scores.get("gtm_score", 5),
        report.overall_viability_score,
        report.investment_readiness_score,
    ]
    colors = ["#48bb78" if v >= 7 else "#ecc94b" if v >= 5 else "#fc8181" for v in vals]

    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{v}/10" for v in vals],
        textposition="outside",
        textfont=dict(color="white")
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,17,23,0.8)",
        font=dict(color="white"),
        xaxis=dict(range=[0, 10.5], tickfont=dict(color="white"), gridcolor="#2d3561"),
        yaxis=dict(tickfont=dict(color="white")),
        height=280,
        margin=dict(l=10, r=60, t=20, b=20)
    )
    return fig


def render_report():
    if not st.session_state.research_complete or not st.session_state.final_state:
        st.warning("No research data yet. Please analyze an idea first.")
        if st.button("Go to Analyze"):
            st.session_state.page = "analyze"
            st.rerun()
        return

    state = st.session_state.final_state
    report = state.get("final_report")
    scores = state.get("scores", {})

    if not report:
        st.error("Report data not found. Please re-run the analysis.")
        return

    # ── Header ──
    st.markdown(f"# 📋 Research Report")
    st.markdown(f"### `{report.startup_idea}`")
    st.markdown("---")

    # ── Score Dashboard ──
    st.markdown("## 📊 Viability Scores")
    mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
    score_items = [
        ("Market", scores.get("market_score", 5)),
        ("Revenue", scores.get("viability_score", 5)),
        ("GTM", scores.get("gtm_score", 5)),
        ("Overall", report.overall_viability_score),
        ("Invest Ready", report.investment_readiness_score),
    ]
    for col, (label, val) in zip([mcol1, mcol2, mcol3, mcol4, mcol5], score_items):
        with col:
            st.metric(label, f"{val}/10", delta=f"{score_to_label(val)}")

    # Charts row
    chart_col1, chart_col2 = st.columns([1, 1])
    with chart_col1:
        st.plotly_chart(render_scores_chart(scores, report), use_container_width=True)
    with chart_col2:
        st.plotly_chart(render_scores_bar(scores, report), use_container_width=True)

    st.markdown("---")

    # ── Tabs for each section ──
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📋 Summary", "📈 Market", "🕵️ Competitors", "⚖️ SWOT", "💰 Revenue", "🚀 GTM", "✅ Recommendations"
    ])

    with tab1:
        st.markdown("### Executive Summary")
        st.markdown(report.executive_summary)

    with tab2:
        st.markdown("### Market Research")
        st.markdown(report.market_research)

    with tab3:
        st.markdown("### Competitor Analysis")
        st.markdown(report.competitor_analysis)

        # Competitor threat visualization
        comp = state.get("competitor_analysis")
        if comp and comp.top_competitors:
            st.markdown("#### Competitor Overview")
            df_comp = pd.DataFrame(comp.top_competitors)
            st.dataframe(df_comp, use_container_width=True)

    with tab4:
        st.markdown("### SWOT Analysis")
        swot = state.get("swot_analysis")
        if swot:
            sw_col, wo_col = st.columns(2)
            with sw_col:
                st.markdown("#### ✅ Strengths")
                for s in swot.strengths:
                    st.markdown(f"- {s}")
                st.markdown("#### ⚠️ Weaknesses")
                for w in swot.weaknesses:
                    st.markdown(f"- {w}")
            with wo_col:
                st.markdown("#### 🌟 Opportunities")
                for o in swot.opportunities:
                    st.markdown(f"- {o}")
                st.markdown("#### 🚨 Threats")
                for t in swot.threats:
                    st.markdown(f"- {t}")
            st.markdown("#### 🧭 Strategic Recommendations")
            for r in swot.strategic_recommendations:
                st.markdown(f"- {r}")
        else:
            st.markdown(report.swot_analysis)

    with tab5:
        st.markdown("### Revenue Strategy")
        st.markdown(report.revenue_strategy)

        rev = state.get("revenue_model")
        if rev and rev.pricing_tiers:
            st.markdown("#### Pricing Tiers")
            df_price = pd.DataFrame(rev.pricing_tiers)
            st.dataframe(df_price, use_container_width=True)

    with tab6:
        st.markdown("### Go-To-Market Plan")
        st.markdown(report.gtm_plan)

        gtm = state.get("gtm_strategy")
        if gtm and gtm.timeline_phases:
            st.markdown("#### Launch Timeline")
            df_phases = pd.DataFrame(gtm.timeline_phases)
            st.dataframe(df_phases, use_container_width=True)

    with tab7:
        st.markdown("### Final Recommendations")
        st.markdown(report.final_recommendations)
        ov = report.overall_viability_score
        if ov >= 7.5:
            st.success(f"🟢 Strong opportunity ({ov}/10) — Proceed with confidence")
        elif ov >= 5.5:
            st.warning(f"🟡 Viable idea ({ov}/10) — Validate key assumptions first")
        else:
            st.error(f"🔴 Needs refinement ({ov}/10) — Revisit core problem/solution fit")

    # ── Downloads ──
    st.markdown("---")
    st.markdown("### 📥 Export Report")
    dcol1, dcol2 = st.columns(2)

    with dcol1:
        try:
            md_path = export_markdown(report)
            with open(md_path, "r", encoding="utf-8") as f:
                md_content = f.read()
            st.download_button(
                "⬇️ Download Markdown Report",
                data=md_content,
                file_name=f"startup_report_{report.startup_idea[:20].replace(' ','_')}.md",
                mime="text/markdown",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Markdown export failed: {e}")

    with dcol2:
        try:
            pdf_path = export_pdf(report)
            if pdf_path:
                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read()
                st.download_button(
                    "⬇️ Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"startup_report_{report.startup_idea[:20].replace(' ','_')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
        except Exception as e:
            st.warning(f"PDF export: {e}")


# ─────────────────────────────────────────────
# PAGE: Architecture
# ─────────────────────────────────────────────
def render_architecture():
    st.title("🏗️ System Architecture")
    st.markdown("---")

    st.markdown("""
    ## Tech Stack
    | Component | Technology | Purpose |
    |-----------|-----------|---------|
    | LLM | Google Gemini 1.5 Flash | Agent reasoning & text generation |
    | Orchestration | LangGraph StateGraph | Multi-agent workflow management |
    | RAG | FAISS + HuggingFace Embeddings | Knowledge retrieval |
    | Web Search | DuckDuckGo + BeautifulSoup | Real-time market intelligence |
    | State Management | Pydantic TypedDict | Type-safe agent communication |
    | Frontend | Streamlit + Plotly | Interactive dashboard |
    | Export | FPDF2 + Markdown | Report generation |

    ## Agent Pipeline
    ```
    User Input (Startup Idea)
         │
         ▼
    ┌─────────────────────┐
    │  LangGraph Graph    │
    │                     │
    │  1. Market Research ──► Web Search + RAG
    │         │
    │  2. Competitor Intel ─► Web Search + RAG
    │         │
    │  3. SWOT Analysis ────► RAG (synthesis)
    │         │
    │  4. Revenue Model ────► Web Search + RAG
    │         │
    │  5. GTM Strategy ─────► Web Search + RAG
    │         │
    │  6. Report Agent ─────► LLM synthesis
    └─────────────────────┘
         │
         ▼
    Final Report + Scores
    ```

    ## RAG Pipeline
    ```
    Knowledge Base (startup frameworks, templates)
         │
         ▼ chunk + embed
    FAISS Vector Index (384-dim vectors)
         │
    Agent Query ──► Embedding ──► Similarity Search ──► Top-K Chunks
         │
         ▼
    Injected into LLM prompt as context
    ```

    ## State Flow
    Each agent reads the full `ResearchState` and writes back a partial update:
    ```python
    def market_research_agent(state: ResearchState) -> dict:
        # Read: state["startup_idea"]
        # Work: web search + RAG + LLM
        # Write: partial state update
        return {"market_research": output, "scores": {...}}
    ```
    """)


# ─────────────────────────────────────────────
# Main App Router
# ─────────────────────────────────────────────
def main():
    init_session_state()
    render_sidebar()

    page = st.session_state.page
    if page == "home":
        render_home()
    elif page == "analyze":
        render_analyze()
    elif page == "report":
        render_report()
    elif page == "architecture":
        render_architecture()
    else:
        render_home()


if __name__ == "__main__":
    main()
