import pymysql
import json

from utils.DB import DB

def retrieve_unrequested_experiments(amount):
    """
    Retrieve up to `amount` experiments from DB that have `sent_requests=0`.
    Return them as a list of dictionaries.
    """
    connection = None
    try:
        connection = DB.get_connection()
        cursor = connection.cursor(pymysql.cursors.DictCursor)

        query = """
            SELECT
                e.exp_id,
                e.based_on,
                e.model_id
            FROM experiment e
            WHERE e.sent_requests=0 
              AND e.state='Waiting'
            LIMIT %s
        """
        cursor.execute(query, (amount,))
        rows = cursor.fetchall()
        return rows

    finally:
        if connection:
            connection.close()


def create_tests_for_experiments(experiment_rows):
    """
    For each experiment row, create a new entry in `test` table, update sent_requests,
    and build the final JSON data for the client.
    Returns a list of dicts, each with {test_id, experiment_data}.
    """
    connection = None
    tests_data = []

    try:
        connection = DB.get_connection()
        cursor = connection.cursor(pymysql.cursors.DictCursor)

        for row in experiment_rows:
            exp_id = row["exp_id"]
            model_id = row["model_id"]

            # 1) Reconstruct the entire experiment data
            experiment_data = build_experiment_dict(cursor, exp_id, model_id)

            # 2) Insert a row into `test` table (assuming columns: test_id, exp_id, date, etc.)
            test_insert_sql = """
                INSERT INTO test (exp_id, date)
                VALUES (%s, NOW())
            """
            cursor.execute(test_insert_sql, (exp_id,))
            test_id = cursor.lastrowid

            # 3) Update experiment so it won't be requested again
            update_sql = """
                UPDATE experiment
                SET sent_requests=1, state='In Progress'
                WHERE exp_id=%s
            """
            cursor.execute(update_sql, (exp_id,))

            # 4) Build the JSON/dict for the client
            tests_data.append({
                "test_id": test_id,
                "exp_id": exp_id,
                "experiment_data": experiment_data
            })

        connection.commit()
        return tests_data

    except Exception as e:
        if connection:
            connection.rollback()
        print("Error in create_tests_for_experiments:", e)
        raise
    finally:
        if connection:
            connection.close()


def build_experiment_dict(cursor, exp_id, model_id):
    """
    Using the given exp_id and model_id, fetch data from 'model' table, 'experiment' table,
    and all layers via 'model_layer' -> 'layer'.
    Return a Python dict that represents the full experiment's data.
    """

    # 1) Fetch experiment info (including based_on)
    experiment_sql = """
        SELECT
            exp_id,
            based_on,
            modification_text,
            state,
            sent_requests
        FROM experiment
        WHERE exp_id=%s
    """
    cursor.execute(experiment_sql, (exp_id,))
    exp_row = cursor.fetchone()

    # 2) Fetch model info (now including epochs)
    model_sql = """
        SELECT
            model_id,
            database_id AS dataset_id,
            loss_fn,
            optimization,
            normalization,
            batch_size,
            weight_decay,
            learning_rate,
            thresh,
            optimization_fields,
            epochs
        FROM model
        WHERE model_id=%s
    """
    cursor.execute(model_sql, (model_id,))
    model_row = cursor.fetchone()

    # 3) Fetch layers via model_layer -> layer
    layers_sql = """
        SELECT
            ml.layer_place,
            ml.out_shape,
            l.*
        FROM model_layer ml
        JOIN layer l ON ml.layer_id = l.layer_id
        WHERE ml.model_id = %s
        ORDER BY ml.layer_place
    """
    cursor.execute(layers_sql, (model_id,))
    layer_rows = cursor.fetchall()

    # Rebuild the layers array
    layers_list = []
    for lr in layer_rows:
        layer_fields = json.loads(lr["layer_fields"]) if lr["layer_fields"] else {}
        layer_input = json.loads(lr["input"]) if lr["input"] else None
        layer_output = json.loads(lr["output"]) if lr["output"] else None

        layers_list.append({
            "layer_type": lr["layer_type"],
            "activation_fn": lr["activation_fn"],
            "weight_initiations": lr["weight_initiations"],
            "input": layer_input,
            "output": layer_output,
            "dropout_rate": lr["dropout_rate"],
            "layer_fields": layer_fields
        })

    # 4) Build the final dictionary
    experiment_dict = {
        "exp_id": exp_id,
        "based_on_id": exp_row["based_on"] or 0,
        "loss_fn": model_row["loss_fn"],
        "optimization": model_row["optimization"],
        "normalization": model_row["normalization"],
        "batch_size": model_row["batch_size"],
        "weight_decay": float(model_row["weight_decay"]),
        "learning_rate": float(model_row["learning_rate"]),
        "epochs": model_row["epochs"],  # include epochs
        "optimization_fields": json.loads(model_row["optimization_fields"]) if model_row["optimization_fields"] else {},
        "layers": layers_list
    }

    return experiment_dict
