import os
import requests
from dotenv import load_dotenv

# Load API key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set. Check your .env file.")


class CustomChatCompletion:
    @staticmethod
    def create(model, messages):
        """
        Mimics the `openai.ChatCompletion.create` method to interact with OpenAI's API.

        :param model: The OpenAI model to use (e.g., 'gpt-4').
        :param messages: List of messages in the format [{'role': 'system', 'content': ...}, {'role': 'user', 'content': ...}].
        :return: The response JSON from OpenAI API.
        """
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": messages,
        }

        try:
            print(payload)
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()  # Raise an error for bad HTTP responses
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error during OpenAI API request: {e}")
            raise
