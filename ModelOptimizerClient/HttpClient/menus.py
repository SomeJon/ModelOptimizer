import os
from HttpClient.client_test_runner import execute_loaded_tests
from HttpClient.fetch_tests import fetch_tests_menu
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



# Entry Point

if __name__ == "__main__":
    main_menu()