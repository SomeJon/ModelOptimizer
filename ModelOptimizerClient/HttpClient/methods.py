import json

from prettytable import PrettyTable
import requests
import os
from dotenv import load_dotenv

# Load .env variables
load_dotenv()
SERVER_URL = os.getenv("SERVER_IP")  # Get the server IP and port from .env


def view_datasets_on_server():
    """
    Fetches and displays datasets available on the server using the `/sql` endpoint.
    """
    try:
        print("Fetching datasets from the server...")

        # SQL query to fetch datasets
        sql_query = "SELECT dataset_id, name, location, train_samples, test_samples, shape, description FROM processed_dataset_data"

        # Send POST request to the `/sql` endpoint
        response = requests.post(
            f"http://{SERVER_URL}/sql",  # Build the URL dynamically from .env
            json={"query": sql_query}
        )

        # Handle response
        if response.status_code == 200:
            response_text = response.text.strip()

            # Remove potential `<pre>` tags if they exist
            if response_text.startswith("<pre>") and response_text.endswith("</pre>"):
                response_text = response_text[5:-6].strip()

            # Split the response text into lines
            lines = response_text.split("\n")
            headers = lines[0].split(" | ")  # Extract headers
            separator_line = lines[1]  # Separator line
            rows = [line.split(" | ") for line in lines[2:]]  # Extract rows

            # Create a PrettyTable to display the data
            table = PrettyTable()
            table.field_names = headers
            for row in rows:
                table.add_row(row)

            print("\nDatasets Available on the Server:\n")
            print(table)
        else:
            print(f"Error fetching datasets: {response.json().get('error', 'Unknown error')}")

    except Exception as e:
        print(f"An error occurred: {e}")


def load_dataset_to_server():
    """
    Load dataset information into the server.
    """
    os.system('cls' if os.name == 'nt' else 'clear')
    print("""
=====================================
       LOAD DATASET TO SERVER
=====================================
    """)

    try:
        # Gather dataset details from the user
        name = input("Enter the Dataset Name: ").strip()
        location = input("Enter the File Location: ").strip()
        train_samples = input("Enter the Number of Training Samples: ").strip()
        test_samples = input("Enter the Number of Testing Samples: ").strip()
        shape = input("Enter the Dataset Shape (e.g., '(28, 28, 3)'): ").strip()
        description = input("Enter the Dataset Description: ").strip()

        # Validate inputs
        if not all([name, location, train_samples, test_samples, shape, description]):
            print("All fields are required. Please try again.")
            input("Click any button to return to the main menu...")
            return

        # Convert sample counts to integers
        try:
            train_samples = int(train_samples)
            test_samples = int(test_samples)
        except ValueError:
            print("Invalid sample count. Both training and testing samples must be numeric values.")
            input("Click any button to return to the main menu...")
            return

        # Display the collected information for confirmation
        print("\nPlease review the dataset information:")
        print(f"Name: {name}")
        print(f"Location: {location}")
        print(f"Number of Training Samples: {train_samples}")
        print(f"Number of Testing Samples: {test_samples}")
        print(f"Shape: {shape}")
        print(f"Description: {description}")
        print("-------------------------------------")
        confirm = input("Enter 1 to confirm or 0 to cancel: ").strip()

        if confirm != "1":
            print("Operation cancelled.")
            input("Click any button to return to the main menu...")
            return

        # Prepare the payload for the server
        dataset_payload = {
            "name": name,
            "location": location,
            "train_samples": train_samples,
            "test_samples": test_samples,
            "shape": shape,
            "description": description
        }

        # Send the request to the server
        print(f"\nUploading dataset '{name}' to the server...")
        response = requests.post(
            f"http://{SERVER_URL}/add_dataset",  # Use the endpoint defined on the server
            json=dataset_payload
        )

        # Handle server response
        if response.status_code == 200 and response.json().get("success", False):
            print("Dataset added successfully!")
        else:
            print(f"Failed to add dataset. Server responded with: {response.text}")

    except Exception as e:
        print(f"An error occurred: {e}")


def generate_new_tests():
    """
    Generate new tests by interacting with the server.
    """
    os.system('cls' if os.name == 'nt' else 'clear')
    print("""
=====================================
        GENERATE NEW TESTS
=====================================
    """)

    try:
        # Reuse the existing method to view datasets on the server
        view_datasets_on_server()

        # User input for dataset ID and generation parameters
        dataset_id = input("\nEnter the Dataset ID: ").strip()
        focus = input("Enter a focus for the tests (e.g., 'Explore simple models'): ").strip()
        num = input("Enter the number of tests to generate: ").strip()
        num_of_based = input("Enter the number of reference experiments to use: ").strip()

        # Validate inputs
        if not dataset_id.isdigit() or not num.isdigit() or not num_of_based.isdigit():
            print("Invalid input. Dataset ID, number of tests, and references must be numeric.")
            input("Press Enter to return to the main menu...")
            return

        dataset_id = int(dataset_id)
        num = int(num)
        num_of_based = int(num_of_based)

        # Confirm user inputs
        print(f"\nGenerating {num} new tests for Dataset ID {dataset_id} with focus: '{focus}' and {num_of_based} reference experiments.")
        confirm = input("Enter 1 to confirm or 0 to cancel: ").strip()

        if confirm != "1":
            print("Operation cancelled.")
            return

        # Choose the model to use
        print("\nChoose the model to use:")
        print("1. GPT-4 Turbo (default)")
        print("2. GPT-3.5 Turbo")
        model_choice = input("Enter your choice (1 or 2): ").strip()

        if model_choice == "2":
            model = "gpt-3.5-turbo"
        else:
            model = "gpt-4-turbo"  # Default to GPT-4 Turbo

        # Send the request to the server
        print("\nSending request to the server...")
        response = requests.post(
            f"http://{SERVER_URL}/gen",
            params={
                "num": num,
                "dataset_id": dataset_id,
                "focus": focus,
                "num_of_based": num_of_based,
                "model": model
            }
        )

        # Handle server response
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                print("\nTests generated successfully! Here are the generated experiments:\n")
                print(json.dumps(data.get("generated_experiments"), indent=2))
            else:
                print(f"Error: {data.get('error', 'Unknown error')}")
        else:
            print(f"Server error: {response.text}")

    except Exception as e:
        print(f"An error occurred: {e}")

