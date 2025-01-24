import os

from HttpClient.methods import *


def main_menu():
    """
    Main menu for the Optimizer client.
    """
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print("""
=====================================
          OPTIMIZER CLIENT
   Efficient Test Management System
=====================================

Main Menu:
1. Fetch Tests to Run
2. Generate New Tests
3. Check Pending Test Count
4. Load Dataset to Server
5. View Local Test Results
6. Execute Loaded Tests
7. Upload Test Results to Server
8. SQL Query to Server
9. View datasets on the server
0. Exit
-------------------------------------
        """)
        choice = input("Please choose an option (1-9): ").strip()

        if choice == "1":
            fetch_tests_menu()
        elif choice == "2":
            generate_new_tests()
        elif choice == "3":
            check_pending_tests()
        elif choice == "4":
            load_dataset_to_server()
        elif choice == "5":
            view_local_test_results()
        elif choice == "6":
            execute_loaded_tests()
        elif choice == "7":
            upload_test_results()
        elif choice == "8":
            sql_query_to_server()
        elif choice == "9":
            view_datasets_on_server()
        elif choice == "0":
            print("Exiting... Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

        input("Click any button to continue...")

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
   - Continuously fetch and run tests.
   - Set a maximum number of tests or run indefinitely.
2. Manual Mode
   - Fetch a specified number of tests and save them to a file.

0. Return to Main Menu
-------------------------------------
        """)
        choice = input("Please choose an option (1-3): ").strip()

        if choice == "1":
            fetch_tests_automatic()
        elif choice == "2":
            fetch_tests_manual()
        elif choice == "0":
            return
        else:
            print("Invalid choice. Please try again.")
            input("Press Enter to continue...")


def check_pending_tests():
    """
    Check how many tests are pending on the server.
    """
    os.system('cls' if os.name == 'nt' else 'clear')
    print("Checking pending tests on the server...")
    # Placeholder for server request
    input("Pending tests: [placeholder result]. Press Enter to return to the main menu...")


def view_local_test_results():
    """
    View local test results stored on the machine.
    """
    os.system('cls' if os.name == 'nt' else 'clear')
    print("Viewing local test results...")
    # Placeholder for file operations
    input("Test results: [placeholder result]. Press Enter to return to the main menu...")

def execute_loaded_tests():
    """
    Execute loaded tests on the local machine.
    """
    os.system('cls' if os.name == 'nt' else 'clear')
    print("""
=====================================
       EXECUTE LOADED TESTS
=====================================
    """)
    print("Options:")
    print("1. Set a maximum number of tests to run")
    print("2. Run all loaded tests")
    print("3. Return to Main Menu")
    print("-------------------------------------")
    choice = input("Please choose an option (1-3): ").strip()

    if choice == "1":
        max_tests = input("Enter the maximum number of tests to run: ").strip()
        print(f"Running up to {max_tests} tests...")
        # Placeholder for execution logic
    elif choice == "2":
        print("Running all loaded tests...")
        # Placeholder for execution logic
    elif choice == "3":
        return
    else:
        print("Invalid choice. Please try again.")
        input("Press Enter to continue...")


def upload_test_results():
    """
    Upload local test results to the server.
    """
    os.system('cls' if os.name == 'nt' else 'clear')
    print("Uploading test results to the server...")
    # Placeholder for server upload
    input("Test results uploaded successfully! Press Enter to return to the main menu...")


def sql_query_to_server():
    """
    Send an SQL query to the server and display results.
    """
    os.system('cls' if os.name == 'nt' else 'clear')
    print("""
=====================================
         SQL QUERY TO SERVER
=====================================
    """)
    query = input("Enter your SQL query: ").strip()
    print("\nSending query to the server...")
    # Placeholder for server request
    print("Query Result: [placeholder result]")
    input("Press Enter to return to the main menu...")

# Fetch Tests Submenu Methods
def fetch_tests_automatic():
    """
    Automatically fetch and run tests from the server.
    """
    print("Automatically fetching and running tests...")
    # Placeholder for automatic fetch logic
    input("Press Enter to return to the Fetch Tests menu...")

def fetch_tests_manual():
    """
    Manually fetch a specific number of tests from the server.
    """
    num_tests = input("Enter the number of tests to fetch: ").strip()
    print(f"Fetching {num_tests} tests...")
    # Placeholder for manual fetch logic
    input("Tests fetched successfully! Press Enter to return to the Fetch Tests menu...")

# Entry Point

if __name__ == "__main__":
    main_menu()