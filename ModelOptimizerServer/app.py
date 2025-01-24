from flask import Flask, request, jsonify

from utils.utils import *


app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello World!'


# SQL execution endpoint
@app.route('/sql', methods=['POST'])
def execute_sql():
    try:
        # Parse SQL query from the request
        data = request.get_json()
        sql_query = data.get("query")

        if not sql_query:
            return jsonify({"error": "No SQL query provided."}), 400

        # Restrict destructive statements
        if any(word in sql_query.upper() for word in ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER"]):
            return jsonify({"error": "Query type not allowed."}), 403

        # Execute the query
        connection = DB.get_connection()
        cursor = connection.cursor()
        cursor.execute(sql_query)
        rows = cursor.fetchall()
        headers = [desc[0] for desc in cursor.description]  # Extract column names
        connection.close()

        # Format the result as an aligned table-like string
        if rows:
            result_table = format_as_aligned_table(headers, rows)
            return f"<pre>{result_table}</pre>", 200  # Use <pre> for HTML formatting
        else:
            return "No results found.", 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/gen', methods=['POST'])
def generate_request():
    """
    Endpoint to create and handle a request JSON for generating tests.
    Decides between generating first-generation tests or tests based on references.
    """
    try:

        # Retrieve query parameters
        num = int(request.args.get('num'))
        dataset_id = int(request.args.get('dataset_id'))
        focus = request.args.get('focus')
        num_of_based = int(request.args.get('num_of_based'))
        model = request.args.get('model', 'gpt-4-turbo')  # Default to gpt-4-turbo if not specified

        # Generate the base JSON
        request_json = create_request_json(num, dataset_id, focus, num_of_based)
        request_data = json.loads(request_json)  # Convert the JSON string back to a dictionary
        if not request_data.get("reference_experiments"):
            # Call first_gen for initial generation
            new_experiments = first_gen(request_data, num, dataset_id, model)
        else:
            # Call gen_request for normal generation
            new_experiments = send_openai_request(request_data, model)
        return jsonify({
            "success": True,
            "generated_experiments": new_experiments
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })



@app.route('/add_dataset', methods=['POST'])
def add_dataset():
    """
    Endpoint to add a dataset to the database.
    Accepts a JSON payload with dataset details.
    """
    try:
        # Parse JSON payload
        data = request.get_json()
        name = data['name']  # Name of the dataset
        location = data['location']  # File path or cloud location
        train_samples = data['train_samples']  # Number of training samples
        test_samples = data['test_samples']  # Number of testing samples
        shape = data['shape']  # Shape of the dataset
        description = data['description']  # Description of the dataset


        # Insert dataset into the database
        connection = DB.get_connection()
        cursor = connection.cursor()
        query = """
            INSERT INTO processed_dataset_data (name, location, train_samples, test_samples, shape, description)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        cursor.execute(query, (name, location, train_samples, test_samples, shape, description))
        connection.commit()

        app.logger.info(f"Dataset '{name}' successfully added to the database.")

        return jsonify({"success": True, "message": "Dataset added successfully!"})

    except Exception as e:
        app.logger.error(f"Error adding dataset: {str(e)}")
        return jsonify({"success": False, "error": str(e)})





if __name__ == '__main__':
    app.run()
