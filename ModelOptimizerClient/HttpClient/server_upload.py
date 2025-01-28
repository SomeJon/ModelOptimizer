import os
import json
import requests
from dotenv import load_dotenv

from HttpClient.client_test_runner import load_tests

SERVER_URL = os.getenv("SERVER_IP")
PENDING_TESTS_FILE = './data/loaded_tests.json'
COMPLETED_TESTS_FILE = './data/loaded_results.json'


def load_environment_variables():
    """
    Loads environment variables from a .env file.
    """
    load_dotenv()  # Defaults to loading from .env in the current directory
    server_ip = os.getenv('SERVER_IP')

    if not server_ip:
        raise ValueError("SERVER_IP must be set in the .env file.")

    return server_ip


def upload_to_server():
    """
    Uploads data to the specified server endpoint.

    Parameters:
    - server_ip (str): Base URL or IP of the server.
    - data (dict or list): JSON-serializable data to upload.
    - endpoint (str): API endpoint path.

    Returns:
    - bool: True if upload was successful, False otherwise.
    """
    if len(load_tests(COMPLETED_TESTS_FILE)) > 0:
        data = load_tests(COMPLETED_TESTS_FILE)
        if upload_json(data):
            clear_loaded_results()
    else:
        print("There are no tests that are ready for upload!")


def clear_loaded_results(file_path=COMPLETED_TESTS_FILE):
    """
    Clears the contents of loaded_results.json by overwriting it with an empty list.

    Parameters:
    - file_path (str): Path to the loaded_results.json file.
    """
    try:
        with open(file_path, 'w') as file:
            json.dump([], file, indent=4)
        print(f"Cleared '{file_path}' after successful upload.")
    except Exception as e:
        print(f"Error clearing '{file_path}': {e}")


def upload_json(json_data):
    url = f"http://{SERVER_URL}/load_results"
    headers = {
        'Content-Type': 'application/json'
    }

    try:
        response = requests.post(url, json=json_data, headers=headers)

        if 200 <= response.status_code < 300:
            print(f"Upload successful: {response.status_code} {response.reason}")
            clear_loaded_results()
            return True
        else:
            print(f"Upload failed: {response.status_code} {response.reason}")
            print(f"Server Response: {response.text}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"Error sending POST request to {url}: {e}")
        return False

