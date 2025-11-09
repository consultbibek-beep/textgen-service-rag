# mini-gen/textgen-service/app.py
import os
from flask import Flask, request, jsonify
from groq import Groq
from dotenv import load_dotenv

# Load .env locally when running outside docker
load_dotenv()

app = Flask(__name__)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    # Keep the app running but calls will return an error if key not present
    app.logger.warning("GROQ_API_KEY not set. Set it in the environment or in the root .env file.")

# Initialize Groq client if possible
groq_client = None
try:
    groq_client = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    app.logger.warning(f"Could not initialize Groq client: {e}")
    groq_client = None

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json(silent=True) or {}
    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "Missing 'prompt' in JSON body"}), 400

    if not GROQ_API_KEY or not groq_client:
        return jsonify({"error": "GROQ API key not set or Groq client unavailable on the server."}), 500

    # System instruction enforces a strict 20-word limit; we also post-process to ensure safety.
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
        # Create chat completion using the llama-3.1-8b-instant model
        completion = groq_client.chat.completions.create(
            messages=messages,
            model="llama-3.1-8b-instant"
        )
        # Extract content
        raw = ""
        try:
            raw = completion.choices[0].message.content
        except Exception:
            # Fallback in case response structure differs
            raw = getattr(completion, "content", "") or str(completion)

        # Post-process: enforce hard 20-word cap (split on whitespace)
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

if __name__ == "__main__":
    # Run on 0.0.0.0 so Docker networking can access it
    app.run(host="0.0.0.0", port=5001)
