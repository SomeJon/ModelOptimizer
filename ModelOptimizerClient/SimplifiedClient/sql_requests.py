import os
import requests
from dotenv import load_dotenv
from prettytable import PrettyTable
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure SERVER_URL is set in your environment variables
load_dotenv()
SERVER_URL = os.getenv('SERVER_IP')  # Default to localhost:5000 if not set


def parse_grid_table(response_text):
    """
    Parses a grid-formatted plain text table and returns headers and rows.
    """
    lines = response_text.strip().split("\n")
    headers = []
    rows = []

    for line in lines:
        line = line.strip()
        if line.startswith('+'):
            continue  # Skip separator lines
        elif line.startswith('|'):
            # Split the line by '|' and strip each cell
            parts = [part.strip() for part in line.strip('|').split('|')]
            if not headers:
                headers = parts  # First '|' line is headers
            else:
                # Handle cases where description contains '|'
                if len(parts) > len(headers):
                    # Merge extra parts into the last column
                    parts = parts[:len(headers) - 1] + [' | '.join(parts[len(headers) - 1:])]
                rows.append(parts)

    return headers, rows


def view_datasets_on_server(ishtml=False):
    """
    Fetches and displays datasets available on the server using the `/sql` endpoint.

    Parameters:
    - ishtml (bool): Determines the format of the response. Defaults to False.
                      If True, expects an HTML table. If False, expects a grid-formatted plain text table.
    """
    try:
        print("Fetching datasets from the server...")

        # SQL query to fetch datasets
        sql_query = """
            SELECT 
                dataset_id, 
                name, 
                location, 
                train_samples, 
                test_samples, 
                shape, 
                description 
            FROM 
                processed_dataset_data
        """

        response_type = "string_table"
        if ishtml:
            response_type = "html"

        response = execute_sql_query(sql_query, response_type)

        # Handle response
        if response:
            response_text = response.text.strip()

            if ishtml:
                # For HTML response, extract the table from <pre> tags
                if response_text.startswith("<pre>") and response_text.endswith("</pre>"):
                    table_html = response_text[5:-6].strip()
                    print("\nDatasets Available on the Server (HTML View):\n")
                    print(table_html)
                else:
                    logging.warning("Expected HTML response wrapped in <pre> tags.")
                    print("\nDatasets Available on the Server (HTML View):\n")
                    print(response_text)
            else:
                # For plain text response, parse the grid-formatted table
                # Optionally, remove <pre> tags if the server includes them
                if response_text.startswith("<pre>") and response_text.endswith("</pre>"):
                    response_text = response_text[5:-6].strip()

                # Check if no results were found
                if response_text.lower() == "no results found.":
                    print("No datasets available on the server.")
                    return

                headers, rows = parse_grid_table(response_text)
                if headers is None or rows is None:
                    print("Failed to parse the dataset table.")
                    return

                # Create a PrettyTable to display the data
                table = PrettyTable()
                table.field_names = headers
                for row in rows:
                    table.add_row(row)

                print("\nDatasets Available on the Server:\n")
                print(table)
        else:
            # Attempt to parse error message from JSON response
            try:
                error_message = response.json().get('error', 'Unknown error.')
            except ValueError:
                error_message = 'Unknown error. Non-JSON response received.'
            print(f"Error fetching datasets: {error_message}")

    except requests.exceptions.Timeout:
        print("Request timed out. Please try again later.")
    except requests.exceptions.ConnectionError:
        print("Failed to connect to the server. Please check the SERVER_URL and server status.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        logging.error("Unexpected error in view_datasets_on_server:", exc_info=True)


def execute_sql_query(sql_query, response_type='data'):
    """
    Executes a SQL query against the server's /sql endpoint.

    Parameters:
    - sql_query (str): The SQL query to execute.
    - response_type (str): The desired response format ('html', 'string_table', 'data').

    Returns:
    - Response content based on the response_type.
    """
    try:
        url = f"{SERVER_URL}/sql"
        params = {'type': response_type}
        headers = {
            'Content-Type': 'application/json'
        }
        payload = {"query": sql_query}

        response = requests.post(
            url,
            params=params,
            json=payload,
            headers=headers,
            timeout=30
        )

        if response.status_code == 200:
            return response
        else:
            try:
                error_message = response.json().get('error', 'Unknown error.')
            except ValueError:
                error_message = 'Unknown error. Non-JSON response received.'
            print(f"Error fetching data: {error_message}")
            return None

    except requests.exceptions.Timeout:
        print("Request timed out. Please try again later.")
        return None
    except requests.exceptions.ConnectionError:
        print("Failed to connect to the server. Please check the SERVER_URL and server status.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        logging.error("Unexpected error in execute_sql_query:", exc_info=True)
        return None
