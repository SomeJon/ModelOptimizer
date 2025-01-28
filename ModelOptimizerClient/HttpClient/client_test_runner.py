import os
import json
import sys

from ModelRunner.runnable_test import train_model, run_tests

# Define the paths to the JSON files
PENDING_TESTS_FILE = './data/loaded_tests.json'
COMPLETED_TESTS_FILE = './data/loaded_results.json'


def display_test_and_result_counts():
    """
    Displays the counts of pending tests and completed test results.
    """
    pending_tests = load_tests(PENDING_TESTS_FILE)
    completed_tests = load_tests(COMPLETED_TESTS_FILE)
    print("\n===== Test and Result Counts =====")
    print(f"Total Pending Tests    : {len(pending_tests)}")
    print(f"Total Completed Tests  : {len(completed_tests)}")
    print("===================================\n")


def load_tests(file_path):
    """
    Loads tests from a JSON file. If the file does not exist, it creates an empty list.

    Parameters:
    - file_path (str): Path to the JSON file.

    Returns:
    - list: List of tests.
    """
    if not os.path.exists(file_path):
        # Initialize with an empty list
        with open(file_path, 'w') as f:
            json.dump([], f, indent=4)
        return []
    with open(file_path, 'r') as f:
        try:
            tests = json.load(f)
            if not isinstance(tests, list):
                raise ValueError(f"The file {file_path} does not contain a JSON array.")
            return tests
        except json.JSONDecodeError:
            print(f"Error: The file {file_path} contains invalid JSON.")
            sys.exit(1)
        except ValueError as ve:
            print(f"Error: {ve}")
            sys.exit(1)


def save_tests(file_path, tests):
    """
    Saves tests to a JSON file.

    Parameters:
    - file_path (str): Path to the JSON file.
    - tests (list): List of tests to save.
    """
    try:
        with open(file_path, 'w') as f:
            json.dump(tests, f, indent=4)
    except Exception as e:
        print(f"Error saving tests to {file_path}: {e}")
        sys.exit(1)


def display_execute_menu():
    """
    Displays the execute submenu options.
    """
    print("\n===== Execute Loaded Tests Submenu =====")
    print("Choose an option:")
    print("1. Run All Pending Tests")
    print("2. Run a Specific Number of Tests")
    print("3. Show Test and Result Counts")
    print("\n0. Return to Main Menu")


def execute_loaded_tests():
    """
    Submenu to execute loaded tests.
    Allows running tests all at once or a specific number, moves tests to completed_results with appropriate status.
    """

    while True:
        # Load pending and completed tests
        pending_tests = load_tests(PENDING_TESTS_FILE)
        completed_tests = load_tests(COMPLETED_TESTS_FILE)

        if not pending_tests:
            print("No pending tests to execute.")
            input("Press Enter to return to the main menu...")
            break

        display_test_and_result_counts()
        # Display the execute submenu
        display_execute_menu()
        choice = input("Enter your choice (0-3): ").strip()

        if choice == '1':
            pending_tests, completed_tests = run_all(pending_tests, completed_tests)

        elif choice == '2':
            print("\nRun a Specific Number of Tests")
            while True:
                num_to_run = input(f"Enter the number of tests to run (1-{len(pending_tests)}): ").strip()
                if num_to_run.lower() == 'all':
                    num = len(pending_tests)
                    break
                if not num_to_run.isdigit():
                    print("Please enter a valid number or type 'all' to run all tests.")
                    continue
                num = int(num_to_run)
                if 1 <= num <= len(pending_tests):
                    break
                else:
                    print(f"Please enter a number between 1 and {len(pending_tests)}.")

            tests_to_run = pending_tests[:num]
            completed_tests, successfully_run_tests = run_tests(tests_to_run, completed_tests)
            # Remove only the executed tests from pending_tests
            if successfully_run_tests:
                # Create a set of IDs for faster lookup
                successfully_run_ids = set(
                    (test.get('exp_id'), test.get('test_id')) for test in successfully_run_tests
                )
                # Filter out the executed tests
                pending_tests = [
                    test for test in pending_tests
                    if (test.get('exp_id'), test.get('test_id')) not in successfully_run_ids
                ]
                print(f"{len(successfully_run_tests)} test(s) have been executed and moved to completed_results.")
            else:
                print("No tests were successfully executed.")

        elif choice == '3':
            # Show Test and Result Counts
            display_test_and_result_counts()

        elif choice == '0':
            # Return to Main Menu
            break

        else:
            print("Invalid choice. Please select an option between 0 and 3.")

        # Save the updated test lists after each operation
        save_tests(PENDING_TESTS_FILE, pending_tests)
        save_tests(COMPLETED_TESTS_FILE, completed_tests)


def run_all(pending_tests, completed_tests):
    print("\nRunning all pending tests...")
    completed_tests, successfully_run_tests = run_tests(pending_tests, completed_tests)
    # After running, remove all tests from pending_tests
    if successfully_run_tests:
        # Create a set of IDs for faster lookup
        successfully_run_ids = set(
            (test.get('exp_id'), test.get('test_id')) for test in successfully_run_tests
        )
        # Remove all tests that have been moved to completed_tests
        pending_tests = [
            test for test in pending_tests
            if (test.get('exp_id'), test.get('test_id')) not in successfully_run_ids
        ]
        print("All executed tests have been moved to completed_results.")
    else:
        print("No tests were successfully executed.")

    return pending_tests, completed_tests
