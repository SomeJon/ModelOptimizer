from flask import Flask, request, jsonify
from utils.generate_requests import *
from utils.load_requests import load_results
from utils.test_requests import *
from utils.utils import *
import sqlparse

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!'


# SQL execution endpoint
@app.route('/sql', methods=['POST'])
def execute_sql():
    try:
        # Parse JSON payload from the request
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON payload."}), 400

        sql_query = data.get("query")
        if not sql_query:
            return jsonify({"error": "No SQL query provided."}), 400

        # Parse and validate the SQL query
        parsed = sqlparse.parse(sql_query)
        if not is_select_query(parsed):
            return jsonify({"error": "Only single SELECT statements are allowed."}), 403

        # Get the 'type' query parameter, defaulting to 'html'
        type_param = request.args.get('type', 'html').lower()
        if type_param not in ['html', 'string_table', 'data']:
            return jsonify({"error": "Invalid type parameter. Allowed values are 'html', 'string_table', 'data'."}), 400

        # Execute the query
        connection = DB.get_connection()
        cursor = connection.cursor()
        cursor.execute(sql_query)

        # Fetch results
        rows = cursor.fetchall()
        headers = [desc[0] for desc in cursor.description]  # Extract column names

        # Close the connection
        cursor.close()
        connection.close()

        # Format and return the results based on 'type' parameter
        if type_param == 'html':
            if rows:
                table_html = format_as_aligned_table(headers, rows, tablefmt="html")
                return f"{table_html}", 200  # 'tabulate' returns HTML table
            else:
                return "No results found.", 200

        elif type_param == 'string_table':
            if rows:
                table_str = format_as_aligned_table(headers, rows, tablefmt="grid")
                return f"{table_str}", 200  # 'tabulate' returns grid-formatted table
            else:
                return "No results found.", 200

        elif type_param == 'data':
            if rows:
                # Convert rows to list of dictionaries
                data_list = [dict(zip(headers, row)) for row in rows]
                return jsonify({"data": data_list}), 200
            else:
                return jsonify({"message": "No results found."}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/gen', methods=['POST'])
def generate_request():
    try:
        # Retrieve query parameters
        num = int(request.args.get('num'))
        dataset_id = int(request.args.get('dataset_id'))
        focus = request.args.get('focus')
        num_of_based = int(request.args.get('num_of_based'))
        model = request.args.get('model', 'gpt-4-turbo')  # default model

        # Prepare the request JSON for generation
        request_json = create_request_json(num, dataset_id, num_of_based)
        request_data = json.loads(request_json)  # parse to dict

        print("Generated request data:", request_data)

        # If no reference_experiments, do a "first_gen" approach; else normal generation
        if not request_data.get("reference_experiments"):
            new_experiments = first_gen(request_data, num, dataset_id, model, focus)
        else:
            new_experiments = send_openai_request(request_data, model, focus)

        # Insert these experiments into DB
        inserted_count = insert_experiments_to_db(new_experiments, dataset_id, focus)

        return jsonify({
            "success": True,
            "message": f"{inserted_count} experiments inserted."
        })
    except Exception as e:
        print("Error in /gen endpoint:", e)
        return jsonify({"success": False, "error": str(e)})


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


@app.route("/get_tests", methods=["GET"])
def get_tests():
    """
    1) Read the desired 'amount' from query params (default to 1 if missing).
    2) Retrieve up to 'amount' experiments with sent_requests=0.
    3) For each, create a corresponding row in 'test' table, update the experiment,
       and build the final JSON for the client.
    4) Return the array of tests with their full experiment data.
    """
    try:
        amount = int(request.args.get("amount", 1))

        # 1) get unrequested experiments
        experiments = retrieve_unrequested_experiments(amount)
        if not experiments:
            return jsonify({
                "success": True,
                "tests": [],
                "message": "No unrequested experiments found"
            })

        # 2) create tests for them
        tests_data = create_tests_for_experiments(experiments)

        return jsonify({
            "success": True,
            "tests": tests_data
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })


@app.route("/get_single_test", methods=["GET"])
def get_single_test():
    try:
        exp_id = int(request.args.get("exp_id", 1))

        experiments = create_test_for_exp(exp_id)
        if not experiments:
            return jsonify({
                "success": True,
                "tests": [],
                "message": "Not found"
            })

        # 2) create tests for them
        tests_data = create_tests_for_experiments(experiments)

        return jsonify({
            "success": True,
            "tests": tests_data
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })


@app.route("/get_single_exp", methods=["GET"])
def get_single_exp():
    connection = None
    try:
        exp_id = int(request.args.get("exp_id", 1))

        experiment = create_test_for_exp(exp_id)
        if not experiment:
            return jsonify({
                "success": True,
                "tests": [],
                "message": "Not found"
            })

        connection = DB.get_connection()
        cursor = connection.cursor(pymysql.cursors.DictCursor)
        exp_id = experiment[0].get("exp_id")
        model_id = experiment[0].get("model_id")

        experiment_data = build_experiment_dict(cursor, exp_id, model_id)

        return jsonify({
            "success": True,
            "tests": experiment_data
        })
    except Exception as e:
        if connection:
            connection.rollback()
        return jsonify({
            "success": False,
            "error": str(e)
        })
    finally:
        if connection:
            connection.close()


@app.route("/count_tests", methods=["GET"])
def count_tests():
    """
    Returns how many tests can currently be fetched, i.e. how many
    experiments have sent_requests=0 (and possibly state='Waiting').
    Example response: { "available": 5 }
    """
    connection = None
    try:
        connection = DB.get_connection()
        cursor = connection.cursor()
        query = """
            SELECT COUNT(*) as cnt
            FROM experiment
            WHERE sent_requests=0
              AND state='Waiting'
        """
        cursor.execute(query)
        row = cursor.fetchone()
        available = row[0] if row else 0

        return jsonify({
            "available": available
        })
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500
    finally:
        if connection:
            connection.close()


@app.route('/load_results', methods=['POST'])
def upload_results():
    data = request.get_json()
    if data is None:
        return jsonify({'error': 'No data received'}), 400

    try:
        # Assuming load_results is a function you'll define later
        load_results(data)
        return jsonify({'message': 'Data received and processed successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run()
