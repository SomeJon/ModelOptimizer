import os
import json
import requests  # or your preferred HTTP library
from dotenv import load_dotenv

# Load .env variables
load_dotenv()
SERVER_URL = os.getenv("SERVER_IP")


def fetch_tests_menu():
    """
    Submenu for fetching tests to run.
    """
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print("""
=====================================
          FETCH TESTS TO RUN
=====================================

1. Automatic Mode
   - Continuously fetch and store 1 test at a time until no more or max reached.
2. Manual Mode
   - Ask how many tests to fetch at once and store them.

3. Check how many tests are available (optional)
0. Return to Main Menu
-------------------------------------
        """)
        choice = input("Please choose an option (1-3, or 0 to return): ").strip()

        if choice == "1":
            fetch_tests_automatic()
        elif choice == "2":
            fetch_tests_manual()
        elif choice == "3":
            check_available_tests()
        elif choice == "0":
            return
        else:
            print("Invalid choice. Please try again.")
            input("Press Enter to continue...")


def fetch_tests_automatic():
    """
    Automatically fetch tests from the server, 1 at a time.
    The user can optionally specify a maximum number.
    If the server returns 0 tests, we stop.
    """
    print("=== AUTOMATIC TEST FETCHING ===")
    max_str = input("Enter maximum tests to fetch (leave blank for infinite): ").strip()
    if max_str == "":
        max_tests = None
    else:
        try:
            max_tests = int(max_str)
            if max_tests <= 0:
                print("No tests will be fetched.")
                input("Press Enter to return...")
                return
        except ValueError:
            print("Invalid number. Returning to menu.")
            input("Press Enter to return...")
            return

    fetched_count = 0
    while True:
        if max_tests is not None and fetched_count >= max_tests:
            print(f"Reached maximum of {max_tests} tests. Stopping.")
            break

        tests = fetch_tests_from_server(amount=1)
        if not tests:
            print("Server returned 0 tests. No more tests available.")
            break

        # We have 1 test (or sometimes the server might return up to 1).
        append_tests_to_file(tests, "./data/loaded_tests")
        fetched_count += len(tests)
        print(f"Fetched {len(tests)} test(s). Total so far: {fetched_count}.")

    print("Automatic fetching complete.")
    input("Press Enter to return to the menu...")


def fetch_tests_manual():
    """
    Manually fetch a specific number of tests from the server, then append to loaded_tests.
    """
    print("=== MANUAL TEST FETCHING ===")
    num_tests_str = input("Enter the number of tests to fetch (0 to cancel): ").strip()
    try:
        num_tests = int(num_tests_str)
        if num_tests <= 0:
            print("Cancelling. No tests will be fetched.")
            input("Press Enter to return...")
            return
    except ValueError:
        print("Invalid number. Returning.")
        input("Press Enter to return...")
        return

    tests = fetch_tests_from_server(amount=num_tests)
    if not tests:
        print("No tests were returned by the server.")
    else:
        append_tests_to_file(tests, "./data/loaded_tests")
        print(f"Fetched {len(tests)} tests and saved to loaded_tests file.")

    input("Press Enter to return to the menu...")


def check_available_tests():
    """
    Optional: Check how many tests are currently available on the server
    if you implement a /count_tests endpoint.
    """
    print("=== CHECK AVAILABLE TESTS ===")
    try:
        # Example GET request to /count_tests endpoint
        resp = requests.get(f"http://{SERVER_URL}/count_tests")
        resp.raise_for_status()
        data = resp.json()
        if "available" in data:
            print(f"Server reports {data['available']} tests can be fetched right now.")
        else:
            print("Server response did not include 'available' field:", data)
    except Exception as e:
        print("Error contacting server:", e)

    input("Press Enter to return to the menu...")

# -------------------------------------------------------------------
# Helper functions


def fetch_tests_from_server(amount):
    """
    Makes a GET request to your server's /get_tests?amount={amount}
    Endpoint should return JSON with structure:
      {
        "success": True/False,
        "tests": [ { test_id, exp_id, experiment_data }, ... ]
      }
    Returns the list of tests (could be empty if none available).
    """
    try:
        url = f"http://{SERVER_URL}/get_tests?amount={amount}"
        resp = requests.get(url)
        resp.raise_for_status()
        data = resp.json()
        if data.get("success") and "tests" in data:
            return data["tests"]
        else:
            print("Server returned success=False or missing 'tests'. Full response:", data)
            return []
    except Exception as e:
        print("Error fetching tests from server:", e)
        return []


def append_tests_to_file(tests, filepath):
    """
    Append the given list of test objects to the specified file, one test per line in JSON.
    Creates the file if it doesn't exist.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    try:
        with open(filepath, "a", encoding="utf-8") as f:
            for t in tests:
                line = json.dumps(t)
                f.write(line + "\n")
        print(f"Appended {len(tests)} tests to {filepath}.")
    except Exception as e:
        print(f"Error writing to file {filepath}:", e)
