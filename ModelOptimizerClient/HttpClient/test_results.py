import os

from dotenv import load_dotenv
from prettytable import PrettyTable
import logging
import json
import requests

from HttpClient.client_test_runner import save_tests
from HttpClient.server_upload import upload_json
from HttpClient.sql_requests import execute_sql_query
from ModelRunner.runnable_test import run_tests

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()
SERVER_URL = os.getenv('SERVER_IP')  # Default to localhost:5000 if not set


def show_all_test_results():
    """
    Option 1: Show all test results where experiment's state != 'Failed', ordered by date.
    Excludes certain fields and includes 'based_on' and 'experiment.date'.
    """
    sql_query = """
        SELECT 
            test.test_id, 
            test.exp_id, 
            test.score, 
            experiment.date AS experiment_date, 
            experiment.based_on, 
            test.date AS test_date, 
            test.hardware_config,
            test.duration_seconds,
            experiment.state
        FROM 
            test
        JOIN 
            experiment ON test.exp_id = experiment.exp_id
        ORDER BY 
            test.date ASC;
    """
    response = execute_sql_query(sql_query, response_type='data')
    if response:
        try:
            response_json = response.json()
            data = response_json.get('data', [])
            if not data:
                print("No test results found.")
                return

            table = PrettyTable()
            # Define the field names based on selected columns
            table.field_names = ["Test ID", "Experiment ID", "Status", "Score", "Experiment Date", "Based On", "Test Date",
                                 "Hardware Config", "Duration (s)"]
            for row in data:
                table.add_row([
                    row.get('test_id', ''),
                    row.get('exp_id', ''),
                    row.get('state', ''),
                    row.get('score', ''),
                    row.get('experiment_date', ''),
                    row.get('based_on', ''),
                    row.get('test_date', ''),
                    row.get('hardware_config', ''),
                    row.get('duration_seconds', '')
                ])
            print("\nAll Test Results:\n")
            print(table)
        except ValueError:
            print("Failed to parse JSON response.")


def show_best_test_results():
    """
    Option 2: Show best test results based on user-specified number, ordered by score.
    Includes 'based_on' and 'experiment.date'.
    """
    while True:
        try:
            n = int(input("Enter the number of top test results to display: "))
            if n <= 0:
                print("Please enter a positive integer.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a numeric value.")

    sql_query = f"""
        SELECT 
            test.test_id, 
            test.exp_id, 
            test.score, 
            experiment.date AS experiment_date, 
            experiment.based_on, 
            test.date AS test_date, 
            test.hardware_config, 
            test.duration_seconds
        FROM 
            test
        JOIN 
            experiment ON test.exp_id = experiment.exp_id
        WHERE 
            experiment.state != 'Failed'
        ORDER BY 
            test.score DESC
        LIMIT {n};
    """
    response = execute_sql_query(sql_query, response_type='data')
    if response:
        try:
            response_json = response.json()
            data = response_json.get('data', [])
            if not data:
                print("No test results found.")
                return

            table = PrettyTable()
            table.field_names = ["Test ID", "Experiment ID", "Score", "Experiment Date", "Based On", "Test Date",
                                 "Hardware Config", "Duration (s)"]
            for row in data:
                table.add_row([
                    row.get('test_id', ''),
                    row.get('exp_id', ''),
                    row.get('score', ''),
                    row.get('experiment_date', ''),
                    row.get('based_on', ''),
                    row.get('test_date', ''),
                    row.get('hardware_config', ''),
                    row.get('duration_seconds', '')
                ])
            print(f"\nTop {n} Test Results by Score:\n")
            print(table)
        except ValueError:
            print("Failed to parse JSON response.")


def show_more_data_of_test_result():
    """
    Option 3: Show more data of a specific test result based on test_id.
    Displays test data score, train data score, run time, and other details.
    """
    while True:
        try:
            test_id = int(input("Enter the Test ID to view more details: "))
            if test_id <= 0:
                print("Please enter a positive integer for Test ID.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a numeric value.")

    sql_query = f"""
        SELECT 
            test.model_architecture, 
            test.score, 
            test.error_message, 
            test.output_loss_epoch, 
            test.output_accuracy_epoch, 
            test.duration_seconds, 
            test.mse, 
            test.variance_dataset, 
            test.variance_y_hat, 
            test.mean_bias, 
            experiment.date AS experiment_date, 
            experiment.based_on
        FROM 
            test
        JOIN 
            experiment ON test.exp_id = experiment.exp_id
        WHERE 
            test.test_id = {test_id};
    """
    response = execute_sql_query(sql_query, response_type='data')
    if response:
        try:
            response_json = response.json()
            data = response_json.get('data', [])
            if not data:
                print(f"No test result found with Test ID {test_id} or the experiment failed.")
                return

            test_data = data[0]
            print(f"\nDetailed Information for Test ID {test_id}:\n")
            print(f"Model Architecture:\n{test_data.get('model_architecture', 'N/A')}\n")
            print(f"Error Message:\n{test_data.get('error_message', 'N/A')}\n")
            print(f"Output Loss per Epoch:\n{test_data.get('output_loss_epoch', 'N/A')}\n")
            print(f"Output Accuracy per Epoch:\n{test_data.get('output_accuracy_epoch', 'N/A')}\n")

            print(f"Test Data Score: {test_data.get('score', 'N/A')}")

            # Extract and display the last value from output_accuracy_epoch as Train Data Score
            output_accuracy_epoch = test_data.get('output_accuracy_epoch', '[]')
            try:
                # Assuming output_accuracy_epoch is a JSON-formatted list
                accuracy_list = json.loads(output_accuracy_epoch)
                if isinstance(accuracy_list, list) and accuracy_list:
                    train_data_score = accuracy_list[-1]
                else:
                    train_data_score = 'N/A'
            except json.JSONDecodeError:
                train_data_score = 'N/A'
            print(f"Train Data Score: {train_data_score}")

            # Display run time
            duration = test_data.get('duration_seconds', 'N/A')
            print(f"Run Time (seconds): {duration}")

            # Additional Tidbits
            print(f"MSE: {test_data.get('mse', 'N/A')}")
            print(f"Variance Dataset: {test_data.get('variance_dataset', 'N/A')}")
            print(f"Variance Y_Hat: {test_data.get('variance_y_hat', 'N/A')}")
            print(f"Mean Bias: {test_data.get('mean_bias', 'N/A')}")
            print(f"Experiment Date: {test_data.get('experiment_date', 'N/A')}")
            print(f"Based On: {test_data.get('based_on', 'N/A')}")
        except ValueError:
            print("Failed to parse JSON response.")


def view_experiment():
    """
    Option 5: View details of a specific experiment based on exp_id.
    """
    while True:
        try:
            exp_id = int(input("Enter the Experiment ID to view details: "))
            if exp_id <= 0:
                print("Please enter a positive integer for Experiment ID.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a numeric value.")

    # Call the /get_single_exp endpoint
    try:
        response = requests.get(f"{SERVER_URL}/get_single_exp", params={'exp_id': exp_id})
        if response.status_code == 200:
            response_json = response.json()
            tests = response_json.get('tests', [])
            if not tests:
                print(f"No experiment found with Experiment ID {exp_id}.")
                return

            # Display experiment details
            print(f"\nDetails for Experiment ID {exp_id}:\n")
            print(json.dumps(tests, indent=4))
        elif response.status_code == 404:
            print(f"No experiment found with Experiment ID {exp_id}.")
        else:
            print(f"Failed to retrieve experiment. Error: {response.json().get('error', 'Unknown Error')}")
    except Exception as e:
        print(f"An error occurred while fetching experiment details: {e}")


def get_and_run_experiment():
    """
    Option 4 (Renamed): Get and Run Experiment based on test_id.
    Currently, it just fetches the experiment JSON without running it.
    """
    while True:
        try:
            exp_id = int(input("Enter the Experiment ID to get and run: "))
            if exp_id <= 0:
                print("Please enter a positive integer for Experiment ID.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a numeric value.")

    # Call the /get_single_test endpoint
    try:
        response = requests.get(f"{SERVER_URL}/get_single_test", params={'exp_id': exp_id})
        if response.status_code == 200:
            response_json = response.json()
            tests = response_json.get('tests', [])
            if not tests:
                print(f"Experiment ID {exp_id} couldn't produce a test!.")
                return

            # Display the JSON data
            print(f"\nTests for Experiment ID {exp_id}:\n")
            results_json = []
            res = run_tests(tests, results_json)
            print(results_json)
            save_tests('./data/loaded_results.json', results_json) #todo delete
            # upload_json(results_json)
        elif response.status_code == 404:
            print(f"No tests found for Experiment ID {exp_id}.")
        else:
            print(f"Failed to retrieve tests. Error: {response.json().get('error', 'Unknown Error')}")
    except Exception as e:
        print(f"An error occurred while fetching tests: {e}")


def test_results():
    """
    Displays the Test Results submenu and handles user interactions.
    """
    while True:
        print("\n--- Test Results Submenu ---")
        print("1. Show All Test Results")
        print("2. Show Best Test Results")
        print("3. Show More Data of a Test Result")
        print("4. Get and Run Experiment")
        print("5. View an Experiment")
        print("0. Back to Main Menu")

        choice = input("Enter your choice: ").strip()

        if choice == '1':
            show_all_test_results()
        elif choice == '2':
            show_best_test_results()
        elif choice == '3':
            show_more_data_of_test_result()
        elif choice == '4':
            get_and_run_experiment()
        elif choice == '5':
            view_experiment()
        elif choice == '0':
            print("Returning to the Main Menu...")
            break
        else:
            print("Invalid choice. Please select a valid option.")
