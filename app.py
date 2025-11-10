# mini-gen-search/textgen-service-rag/app.py (Consolidated API)
from flask import Flask
import os
import logging
from dotenv import load_dotenv

# Import the logic modules
# FIX: Removed the non-existent function 'run_simple_generation_app'
from textgen_api import setup_simple_generation_route
from rag_api import setup_rag_route, ingest_pdf, setup_rag

load_dotenv()
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app.logger.info("Starting combined TextGen/RAG API...")

# --- Setup Simple Generation API (Original Functionality) ---
# Use the function from textgen_api.py to register the /generate route
setup_simple_generation_route(app)
app.logger.info("✅ Simple LLM Generation route (/generate) setup complete.")

# --- Setup RAG API (New Functionality) ---
vector_store = ingest_pdf(app.logger)
rag_chain = setup_rag(vector_store, app.logger)
# Use the function from rag_api.py to register the /ask route
setup_rag_route(app, rag_chain)

# --- Health Check ---
@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check."""
    return {"status": "ok", "simple_llm_ready": True, "rag_ready": rag_chain is not None}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)