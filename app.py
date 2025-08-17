import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import base64

# --- Configuration ---

# Load API key from environment variable for security.
# IMPORTANT: DO NOT hardcode your API key here.
# You MUST set this environment variable in your hosting platform (e.g., Render, Vercel, Heroku).
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    # If the API key is not set, raise an error to prevent deployment issues.
    raise ValueError("GEMINI_API_KEY environment variable not set. Please configure it in your hosting environment.")

genai.configure(api_key=GEMINI_API_KEY)

app = Flask(__name__)
# Enable CORS for all origins. For production, consider restricting this
# to your Framer app's specific domain (e.g., origins=["https://your-framer-app.framer.app"]).
CORS(app)

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

# System prompt for medical image analysis
system_prompt = """Your Responsibilities include:

1. Detailed Analysis: Thoroughly analyze each image, focusing on identifying any abnormal findings.

2. Findings Report: Document all observed anomalies or signs of disease. Clearly articulate these findings in a structured format.

3. Recommendations and Next Steps: Based on your analysis, suggest potential next steps, including further tests or treatments as applicable.

4. Treatment Suggestions: If appropriate, recommend possible treatment options or interventions.

Important Notes:

1. Scope of Response: Only respond if the image pertains to human health issues.

2. Clarity of Image: In cases where the image quality impedes clear analysis, note that certain aspects are 'Unable to be determined based on the provided image.'

3. Disclaimer: Accompany your analysis with the disclaimer: "Consult with a Doctor before making any decisions."

4. Your insights are invaluable in guiding clinical decisions. Please proceed with the analysis, adhering to the structured approach outlined above.
"""

# --- API Endpoint ---

@app.route('/analyze-image', methods=['POST'])
def analyze_image():
    """
    Receives base64 image data and mimeType from the frontend,
    sends it to the Gemini API, and returns the analysis.
    """
    if not request.json:
        return jsonify({"error": "Request must be JSON"}), 400

    image_data_base64 = request.json.get('imageData')
    mime_type = request.json.get('mimeType')

    if not image_data_base64 or not mime_type:
        return jsonify({"error": "Missing imageData or mimeType in request body"}), 400

    try:
        # Construct the parts for the Gemini API call
        image_parts = [{"mime_type": mime_type, "data": image_data_base64}]
        prompt_parts = [image_parts[0], system_prompt]

        # Initialize the Generative Model
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash-latest",
            generation_config=generation_config,
            safety_settings=safety_setting
        )

        # Generate content from the model
        response = model.generate_content(prompt_parts)

        # Check if the response contains text analysis
        if response and response.text:
            return jsonify({"analysis": response.text}), 200
        else:
            # Provide more details if the model blocks content or returns no text
            error_details = "No text in response."
            if response and response.prompt_feedback:
                error_details = f"Prompt feedback: {response.prompt_feedback}"
            return jsonify({"error": "Failed to get analysis from Gemini API", "details": error_details}), 500

    except Exception as e:
        # Log the detailed backend error for debugging purposes
        print(f"Backend Error during Gemini API call: {e}")
        # Handle specific Gemini API errors if needed (e.g., quota exceeded)
        if "ResourceExhausted" in str(e):
            return jsonify({"error": "Quota exceeded. Please try again later.", "backend_detail": str(e)}), 429
        return jsonify({"error": f"An unexpected error occurred on the backend: {str(e)}", "backend_detail": str(e)}), 500

# --- Server Start ---

if __name__ == '__main__':
    # For local development, run with debug=True.
    # For production deployment, use a production-ready WSGI server like Gunicorn.
    # The PORT environment variable is typically provided by hosting platforms.
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=os.environ.get('FLASK_DEBUG') == '1', host='0.0.0.0', port=port)
