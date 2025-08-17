import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai

# --- Configuration ---

# Load API key from environment variable for security.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set. Please configure it in your hosting environment.")

genai.configure(api_key=GEMINI_API_KEY)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Model configuration for Gemini API
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
}

safety_setting = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

system_prompt = """Your Responsibilities include:

1. Detailed Analysis: Thoroughly analyze each image, focusing on identifying any abnormal findings.
2. Findings Report: Document all observed anomalies or signs of disease. Clearly articulate these findings in a structured format.
3. Recommendations and Next Steps: Based on your analysis, suggest potential next steps, including further tests or treatments as applicable.
4. Treatment Suggestions: If appropriate, recommend possible treatment options or interventions.

Important Notes:
- Only respond if the image pertains to human health issues.
- If image quality is unclear, say 'Unable to be determined based on the provided image.'
- Always include disclaimer: "Consult with a Doctor before making any decisions."
"""

# --- Routes ---

@app.route("/")
def home():
    """Health check route so Render doesn’t show 404"""
    return jsonify({"status": "ok", "message": "Medical Image Analyzer Backend is running ✅"}), 200

@app.route("/analyze-image", methods=["POST"])
def analyze_image():
    if not request.json:
        return jsonify({"error": "Request must be JSON"}), 400

    image_data_base64 = request.json.get("imageData")
    mime_type = request.json.get("mimeType")

    if not image_data_base64 or not mime_type:
        return jsonify({"error": "Missing imageData or mimeType in request body"}), 400

    try:
        image_parts = [{"mime_type": mime_type, "data": image_data_base64}]
        prompt_parts = [image_parts[0], system_prompt]

        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash-latest",
            generation_config=generation_config,
            safety_settings=safety_setting,
        )

        response = model.generate_content(prompt_parts)

        if response and response.text:
            return jsonify({"analysis": response.text}), 200
        else:
            error_details = "No text in response."
            if response and response.prompt_feedback:
                error_details = f"Prompt feedback: {response.prompt_feedback}"
            return jsonify({"error": "Failed to get analysis from Gemini API", "details": error_details}), 500

    except Exception as e:
        print(f"Backend Error: {e}")
        if "ResourceExhausted" in str(e):
            return jsonify({"error": "Quota exceeded. Try again later.", "backend_detail": str(e)}), 429
        return jsonify({"error": f"Unexpected backend error: {str(e)}", "backend_detail": str(e)}), 500


# --- Server Start ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1", host="0.0.0.0", port=port)

