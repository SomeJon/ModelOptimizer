import os
import json
from datetime import datetime
import sys
from ModelRunner.runnable_test import train_model
from ModelRunner.get_cifar_dataset import get_cifar10_datasets

PENDING_TESTS_FILE = './data/loaded_results'
COMPLETED_TESTS_FILE = './data/loaded_tests'


def load_tests(file_path):
    """
    Loads tests from a JSON file. If the file does not exist, it creates an empty list.

    Parameters:
    - file_path (str): Path to the JSON file.

    Returns:
    - list: List of tests.
    """
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            json.dump([], f)
        return []
    with open(file_path, 'r') as f:
        try:
            tests = json.load(f)
            if not isinstance(tests, list):
                raise ValueError(f"The file {file_path} does not contain a list.")
            return tests
        except json.JSONDecodeError:
            print(f"Error: The file {file_path} contains invalid JSON.")
            sys.exit(1)


def save_tests(file_path, tests):
    """
    Saves tests to a JSON file.

    Parameters:
    - file_path (str): Path to the JSON file.
    - tests (list): List of tests to save.
    """
    with open(file_path, 'w') as f:
        json.dump(tests, f, indent=4)


def display_menu(total_tests, completed_tests):
    """
    Displays the submenu with current test statistics.

    Parameters:
    - total_tests (int): Number of pending tests.
    - completed_tests (int): Number of completed tests.
    """
    print("\n===== Test Manager Submenu =====")
    print(f"Total Pending Tests   : {total_tests}")
    print(f"Completed Tests      : {completed_tests}\n")
    print("Choose an option:")
    print("1. Run All Pending Tests")
    print("2. Run a Specific Number of Tests")
    print("3. Exit")


def run_tests(selected_tests, completed_tests):
    """
    Executes the selected tests and updates the completed_tests list.

    Parameters:
    - selected_tests (list): List of tests to execute.
    - completed_tests (list): List to append the results of executed tests.

    Returns:
    - list: Updated completed_tests list.
    """
    for idx, test in enumerate(selected_tests, 1):
        print(f"\nExecuting Test {idx}/{len(selected_tests)}:")
        exp_id = test.get('exp_id', 'Unknown')
        test_id = test.get('test_id', 'Unknown')
        print(f"Exp ID: {exp_id}, Test ID: {test_id}")

        # Extract necessary information from the test configuration
        json_str = json.dumps(test.get('experiment_data', {}))

        # Load CIFAR-10 datasets
        train_dataset, test_dataset = get_cifar10_datasets(data_dir='./data')

        # Execute the training process
        result_json = train_model(json_str, train_dataset, test_dataset)

        # Parse the result JSON
        try:
            result = json.loads(result_json)
            # Add a timestamp to the result
            result['execution_timestamp'] = datetime.now().isoformat()
            completed_tests.append(result)
            print(f"Test {exp_id}-{test_id} completed successfully.")
        except json.JSONDecodeError:
            print(f"Error: Failed to parse the result of Test {exp_id}-{test_id}.")

    return completed_tests


def main():
    """
    Main function to manage and execute tests.
    """
    # Load pending and completed tests
    pending_tests = load_tests(PENDING_TESTS_FILE)
    completed_tests = load_tests(COMPLETED_TESTS_FILE)

    while True:
        total_tests = len(pending_tests)
        num_completed = len(completed_tests)
        display_menu(total_tests, num_completed)

        choice = input("Enter your choice (1-3): ").strip()

        if choice == '1':
            if total_tests == 0:
                print("No pending tests to run.")
                continue
            print("\nRunning all pending tests...")
            completed_tests = run_tests(pending_tests, completed_tests)
            # After running, clear the pending_tests
            pending_tests = []
            print("All pending tests have been executed.")

        elif choice == '2':
            if total_tests == 0:
                print("No pending tests to run.")
                continue
            while True:
                num_to_run = input(f"Enter the number of tests to run (1-{total_tests}): ").strip()
                if num_to_run.lower() == 'all':
                    num = total_tests
                    break
                if not num_to_run.isdigit():
                    print("Please enter a valid number.")
                    continue
                num = int(num_to_run)
                if 1 <= num <= total_tests:
                    break
                else:
                    print(f"Please enter a number between 1 and {total_tests}.")

            tests_to_run = pending_tests[:num]
            completed_tests = run_tests(tests_to_run, completed_tests)
            # Remove the executed tests from pending_tests
            pending_tests = pending_tests[num:]
            print(f"{num} test(s) have been executed and moved to completed_tests.")

        elif choice == '3':
            # Save the updated test lists before exiting
            save_tests(PENDING_TESTS_FILE, pending_tests)
            save_tests(COMPLETED_TESTS_FILE, completed_tests)
            print("Exiting Test Manager. Goodbye!")
            break

        else:
            print("Invalid choice. Please select an option between 1 and 3.")

        # Save the updated test lists after each operation
        save_tests(PENDING_TESTS_FILE, pending_tests)
        save_tests(COMPLETED_TESTS_FILE, completed_tests)


if __name__ == "__main__":
    main()
