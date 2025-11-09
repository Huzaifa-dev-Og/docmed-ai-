import os
from dotenv import load_dotenv
import json
from flask import Flask, jsonify, request
from flask_cors import CORS
from google import genai
from google.genai import types 

# --- Initialization ---
load_dotenv()
# Note: The GEMINI_API_KEY must be set in a .env file for this to run locally.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("FATAL ERROR: GEMINI_API_KEY not found. Exiting.")
    exit(1)

# Initialize the Gemini Client
try:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
except Exception as e:
    print(f"Error initializing Gemini client: {e}")
    exit(1)

app = Flask(__name__)
# Crucial: Allows the frontend to access this server on port 5000
CORS(app) 

# --- Medical Information Schema and Prompt ---

def get_medical_schema() -> types.Schema:
    """Defines the structured output for general medical information."""
    return types.Schema(
        type=types.Type.OBJECT,
        properties={
            "query": types.Schema(
                type=types.Type.STRING,
                description="The original user query."
            ),
            "potential_condition": types.Schema(
                type=types.Type.STRING,
                description="The general condition or topic related to the query."
            ),
            "summary": types.Schema(
                type=types.Type.STRING,
                description="A concise educational summary of the topic."
            ),
            "common_symptoms": types.Schema(
                type=types.Type.ARRAY,
                items=types.Schema(type=types.Type.STRING),
                description="A list of common symptoms associated with the condition."
            ),
            "recommended_actions": types.Schema(
                type=types.Type.ARRAY,
                items=types.Schema(type=types.Type.STRING),
                description="General advice for next steps, such as resting or consulting a doctor."
            ),
            "important_disclaimer": types.Schema(
                type=types.Type.STRING,
                description="A mandatory safety warning that this is AI medical advice But go to the doctor for more appropriate answers."
            )
        },
        required=["query", "potential_condition", "summary", "important_disclaimer"]
    )

def create_medical_prompt(user_input: str) -> str:
    """Creates the system prompt with a safety disclaimer."""
    return f"""
    You are a non-diagnostic medical information assistant. Your goal is to provide general, solution, educational information based on a user's health-related query.

    **CRITICAL SAFETY WARNING:** You must include a prominent and specific legal disclaimer in the 'important_disclaimer' field. This information is NOT medical advice, diagnosis, or treatment and should not replace consultation with a qualified healthcare professional.

    Based on the user's input: "{user_input}". Generate a detailed, educational response strictly in a single JSON block following the provided schema.
    """


# --- API Route ---

@app.route('/', methods=['GET'])
def home():
    """Simple check to confirm the server is running."""
    return jsonify({"message": "Medical Assistant Backend is running. Endpoint: /api/get-medical-info"}), 200

@app.route('/api/get-medical-info', methods=['POST'])
def get_medical_info():
    """Generates structured medical information based on user prompt."""
    data = request.get_json()
    user_prompt = data.get('prompt', '').strip()

    if not user_prompt:
        return jsonify({"error": "Missing 'prompt' in request body."}), 400
    
    try:
        system_prompt = create_medical_prompt(user_prompt)
        
        # Configuration for structured JSON output
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=get_medical_schema()
        )
        
        # Call the Gemini API
        gemini_response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[system_prompt],
            config=config,
        )

        # Parse the JSON response
        response_data = json.loads(gemini_response.text)
        
        return jsonify({
            "success": True,
            "data": response_data
        }), 200

    except Exception as e:
        # Handle API or JSON parsing errors
        print(f"Error during API call or processing: {e}")
        return jsonify({"error": f"Failed to get medical information. Detail: {e}"}), 500

if __name__ == '__main__':
    # Medical assistant backend will run on port 5000
    app.run(port=5000, debug=True)
