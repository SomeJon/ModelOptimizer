import os

from dotenv import load_dotenv
from prettytable import PrettyTable
import logging
import json
import requests

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
            test.bias,
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
            table.field_names = ["Test ID", "Experiment ID", "Status", "Score", "Bias", "Based On", "Test Date",
                                 "Experiment Date", "Hardware Config", "Duration (s)"]
            for row in data:
                table.add_row([
                    row.get('test_id', ''),
                    row.get('exp_id', ''),
                    row.get('state', ''),
                    row.get('score', ''),
                    row.get('bias', ''),
                    row.get('based_on', ''),
                    row.get('test_date', ''),
                    row.get('experiment_date', ''),
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
            test.bias,
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
            table.field_names = ["Test ID", "Experiment ID", "Score", "Bias", "Based On", "Test Date", "Experiment Date",
                                 "Hardware Config", "Duration (s)"]
            for row in data:
                table.add_row([
                    row.get('test_id', ''),
                    row.get('exp_id', ''),
                    row.get('score', ''),
                    row.get('bias', ''),
                    row.get('based_on', ''),
                    row.get('test_date', ''),
                    row.get('experiment_date', ''),
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
    Displays comprehensive details including model architecture, scores, losses, accuracies, run time, and additional statistics.
    """
    import json
    from tabulate import tabulate

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
            test.epoch_losses_train, 
            test.epoch_losses_validation, 
            test.epoch_accuracies_train, 
            test.epoch_accuracies_validation, 
            test.duration_seconds, 
            test.bias, 
            test.test_loss, 
            test.epochs_trained, 
            test.date AS test_done_date,
            experiment.date AS experiment_date, 
            experiment.based_on,
            experiment.modification_text
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
                print(f"\nNo test result found with Test ID {test_id} or the experiment failed.\n")
                return

            test_data = data[0]

            # Define a helper function for safe data retrieval
            def get_field(field_name, default='N/A'):
                return test_data.get(field_name, default)

            # Formatting helper
            def format_section(title, content, separator="=" * 50):
                print(f"\n{separator}")
                print(f"{title}")
                print(f"{separator}")
                print(content)
                print(separator)

            # Model Information
            model_architecture = get_field('model_architecture')
            error_message = get_field('error_message')

            model_info = f"Model Architecture:\n{model_architecture}\n"
            error_info = f"Error Message:\n{error_message}\n" if error_message != 'N/A' else "Error Message: None\n"

            format_section("Model Information", model_info + error_info)

            # Training and Validation Metrics
            epoch_losses_train = get_field('epoch_losses_train')
            epoch_losses_validation = get_field('epoch_losses_validation')
            epoch_accuracies_train = get_field('epoch_accuracies_train')
            epoch_accuracies_validation = get_field('epoch_accuracies_validation')

            # Convert JSON strings to Python lists if necessary
            try:
                epoch_losses_train = json.loads(epoch_losses_train) if isinstance(epoch_losses_train,
                                                                                  str) else epoch_losses_train
            except json.JSONDecodeError:
                epoch_losses_train = 'Invalid JSON format'

            try:
                epoch_losses_validation = json.loads(epoch_losses_validation) if isinstance(epoch_losses_validation,
                                                                                            str) else epoch_losses_validation
            except json.JSONDecodeError:
                epoch_losses_validation = 'Invalid JSON format'

            try:
                epoch_accuracies_train = json.loads(epoch_accuracies_train) if isinstance(epoch_accuracies_train,
                                                                                          str) else epoch_accuracies_train
            except json.JSONDecodeError:
                epoch_accuracies_train = 'Invalid JSON format'

            try:
                epoch_accuracies_validation = json.loads(epoch_accuracies_validation) if isinstance(
                    epoch_accuracies_validation, str) else epoch_accuracies_validation
            except json.JSONDecodeError:
                epoch_accuracies_validation = 'Invalid JSON format'

            # Prepare data for table
            epochs_trained = get_field('epochs_trained', 0)
            table_data = []
            for epoch in range(1, epochs_trained + 1):
                train_loss = epoch_losses_train[epoch - 1] if isinstance(epoch_losses_train, list) and len(
                    epoch_losses_train) >= epoch else 'N/A'
                valid_loss = epoch_losses_validation[epoch - 1] if isinstance(epoch_losses_validation, list) and len(
                    epoch_losses_validation) >= epoch else 'N/A'
                train_acc = epoch_accuracies_train[epoch - 1] if isinstance(epoch_accuracies_train, list) and len(
                    epoch_accuracies_train) >= epoch else 'N/A'
                valid_acc = epoch_accuracies_validation[epoch - 1] if isinstance(epoch_accuracies_validation,
                                                                                 list) and len(
                    epoch_accuracies_validation) >= epoch else 'N/A'

                table_data.append([epoch, train_loss, valid_loss, train_acc, valid_acc])

            table_headers = ["Epoch", "Train Loss", "Validation Loss", "Train Accuracy", "Validation Accuracy"]
            training_metrics_table = tabulate(table_data, headers=table_headers, tablefmt="pretty")

            format_section("Training and Validation Metrics per Epoch", training_metrics_table)

            epochs_trained = get_field('epochs_trained')
            bias = get_field('bias')
            test_loss = get_field('test_loss')
            # Scores and Run Time
            score = get_field('score')
            # Extract and display the last value from epoch_accuracies_train as Train Data Score
            train_data_score = epoch_accuracies_train[-1] if isinstance(epoch_accuracies_train,
                                                                        list) and epoch_accuracies_train else 'N/A'
            duration_seconds = get_field('duration_seconds')
            experiment_date = get_field('experiment_date')
            based_on = get_field('based_on')
            test_done_date = get_field('test_done_date')
            modification_text = get_field('modification_text')

            additional_stats = f"Date of test: {test_done_date}\n" \
                               f"It is based on: {based_on}\n" \
                               f"Experiment creation date: {experiment_date}\n" \
                               f"Running Durations(s): {duration_seconds}\n" \
                               f"Score: {score}\n" \
                               f"Train data score: {train_data_score}\n" \
                               f"Test loss: {test_loss}\n" \
                               f"Bias: {bias}\n" \
                               f"Focus: {modification_text}\n" \
                               f"epochs trained: {epochs_trained}\n"

            format_section("Statistics", additional_stats)

        except ValueError:
            print("\nFailed to parse JSON response.\n")


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
            upload_json(results_json)
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
