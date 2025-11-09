# mini-gen-search/textgen-service-rag/textgen_api.py
import os
from flask import request, jsonify
from groq import Groq
from dotenv import load_dotenv

# Load .env locally when running outside docker
load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

groq_client = None
try:
    if GROQ_API_KEY:
        groq_client = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    print(f"Could not initialize Groq client: {e}") # Use print/logging outside of Flask app context

def setup_simple_generation_route(app):
    """Registers the /generate route to the Flask app."""

    @app.route("/generate", methods=["POST"])
    def generate():
        data = request.get_json(silent=True) or {}
        prompt = data.get("prompt", "").strip()
        if not prompt:
            return jsonify({"error": "Missing 'prompt' in JSON body"}), 400

        if not GROQ_API_KEY or not groq_client:
            return jsonify({"error": "GROQ API key not set or Groq client unavailable on the server."}), 500

        system_instructions = (
            "You are a concise assistant. Produce an answer that is AT MOST 20 WORDS long. "
            "Do not include extraneous explanations, lists, or surrounding punctuation. "
            "If the user's prompt requires more than 20 words for full accuracy, produce a concise 20-word summary."
        )

        messages = [
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": prompt},
        ]

        try:
            completion = groq_client.chat.completions.create(
                messages=messages,
                model="llama-3.1-8b-instant"
            )
            raw = completion.choices[0].message.content

            words = raw.strip().split()
            if len(words) > 20:
                truncated = " ".join(words[:20])
                generated = truncated
            else:
                generated = raw.strip()

            return jsonify({"generated": generated}), 200

        except Exception as e:
            app.logger.exception("Error calling Groq API")
            return jsonify({"error": f"Exception while calling Groq API: {str(e)}"}), 500