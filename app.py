"""
app.py
======
Entry point for the Streamlit application.

Run with:
    streamlit run app.py

This file:
1. Sets up logging
2. Initializes the RAG knowledge base (FAISS index)
3. Launches the Streamlit dashboard

DEPLOYMENT:
For Streamlit Cloud, this is the file you point to.
For local dev: streamlit run app.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Validate critical environment variables before launching
if not os.getenv("GEMINI_API_KEY"):
    print("ERROR: GEMINI_API_KEY not set. Copy .env.example to .env and add your key.")
    print("Get a free key at: https://aistudio.google.com")
    sys.exit(1)

# Set up logging
os.makedirs("logs", exist_ok=True)
from utils.logger import setup_logger, get_logger
setup_logger(os.getenv("LOG_LEVEL", "INFO"))
logger = get_logger(__name__)

logger.info("Starting Multi-Agent Startup Research Assistant")

# Pre-build FAISS index on startup (so first query is fast)
try:
    from rag.vector_store import get_vector_store
    logger.info("Initializing RAG knowledge base...")
    vs = get_vector_store()
    logger.info("RAG knowledge base ready")
except Exception as e:
    logger.warning(f"RAG initialization warning: {e}. Will retry on first use.")

# Launch Streamlit dashboard
from frontend.dashboard import main
main()
