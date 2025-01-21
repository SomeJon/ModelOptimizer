from flask import Flask, request, jsonify
from dotenv import load_dotenv
import pymysql
import os

from utils.utils import format_as_aligned_table

load_dotenv()

app = Flask(__name__)

def get_db_connection():
    connection = pymysql.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        port=3306
    )
    return connection

@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/report/', methods=['POST'])
def report_results():
    try:
        data = request.get_json()
        connection = get_db_connection()
        cursor = connection.cursor()
        cursor.execute("""
            INSERT INTO test (exp_id, score, duration, date, hardware_config)
            VALUES (%s, %s, %s, NOW(), %s)
        """, (data['exp_id'], data['score'], data['duration'], data['hardware_config']))
        connection.commit()
        connection.close()
        return "Test results reported successfully!", 200
    except Exception as e:
        return f"Error: {str(e)}", 500


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
        connection = get_db_connection()
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

if __name__ == '__main__':
    app.run()
