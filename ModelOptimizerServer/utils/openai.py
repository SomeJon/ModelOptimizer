import openai
import os
from dotenv import load_dotenv
import json


# Load API key from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def send_openai_request(request_json):
    try:
        openai.api_key = OPENAI_API_KEY

        # Send the request to OpenAI
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates machine learning experiments."},
                {
                    "role": "user",
                    "content": (
                        "The following is a JSON request to generate machine learning experiments.\n"
                        "Please return only valid JSON, strictly matching the format provided in the input.\n"
                        "Do not include explanations or extra text.\n"
                        f"Request JSON:\n{request_json}"
                    )
                }
            ]
        )

        # Parse the response
        result = response['choices'][0]['message']['content']

        # Validate JSON
        try:
            json_result = json.loads(result)  # Ensures the response is valid JSON
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON returned by OpenAI: {e}")

        return json_result

    except Exception as e:
        print(f"Error during OpenAI request: {e}")
        raise
