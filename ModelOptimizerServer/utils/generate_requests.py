import json
import pymysql
from utils.DB import DB
from utils.openai import send_openai_request


def remove_outer_quotes(s):
    """
    Removes outer single or double quotes from a string if present.

    :param s: The input string.
    :return: The string without outer quotes.
    """
    if isinstance(s, str) and len(s) >= 2:
        if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
            return s[1:-1]
    return s


def create_request_json(num, dataset_id, focus, num_of_based):
    """
    Create a request JSON for generating new experiments.

    :param num: Number of tests to generate.
    :param dataset_id: The ID of the dataset to base the tests on.
    :param focus: Instructions for the focus of the tests.
    :param num_of_based: Number of top experiments to use as references.
    :return: Dictionary with the request structure for OpenAI.
    """
    connection = None
    try:
        # Establish DB connection
        connection = DB.get_connection()
        cursor = connection.cursor(pymysql.cursors.DictCursor)

        # Step 1: Fetch top 'num_of_based' experiments
        query_top_experiments = """
            SELECT exp.exp_id
            FROM experiment exp
            INNER JOIN model m ON exp.model_id = m.model_id
            INNER JOIN (
                SELECT exp_id, MAX(score) AS max_score
                FROM test
                GROUP BY exp_id
            ) t_max ON exp.exp_id = t_max.exp_id
            INNER JOIN test t ON exp.exp_id = t.exp_id AND t.score = t_max.max_score
            WHERE m.database_id = %s
            ORDER BY t.score DESC
            LIMIT %s;
        """
        cursor.execute(query_top_experiments, (dataset_id, num_of_based))
        top_experiments = cursor.fetchall()

        print(f"Number of top experiments fetched: {len(top_experiments)}")  # Debug Statement

        if not top_experiments:
            print("No top experiments found for the given dataset_id and num_of_based.")
            return {
                "instructions": {
                    "count": num,
                    "focus": focus,
                    "note": "In the result JSONs, replace exp_id with based_on_id."
                },
                "options": {
                    "loss_fn": [],
                    "optimization": [],
                    "normalization": [],
                    "layer_types": {},
                    "optimization_fields": {}
                },
                "reference_experiments": []
            }

        # Step 2: For each top experiment, fetch detailed info including layers
        reference_experiments = []
        for exp in top_experiments:
            exp_id = exp["exp_id"]

            query_experiment_details = """
                SELECT exp.*, m.*, ml.layer_id, ml.layer_place, l.*
                FROM experiment exp
                INNER JOIN model m ON exp.model_id = m.model_id
                INNER JOIN model_layer ml ON m.model_id = ml.model_id
                INNER JOIN layer l ON ml.layer_id = l.layer_id
                WHERE m.database_id = %s AND exp.exp_id = %s
                ORDER BY ml.layer_place ASC;
            """
            cursor.execute(query_experiment_details, (dataset_id, exp_id))
            experiment_rows = cursor.fetchall()

            if not experiment_rows:
                print(f"No details found for experiment_id {exp_id}. Skipping.")
                continue

            # Organize layers
            layers = []
            for row in experiment_rows:
                layer = {
                    "layer_id": row["layer_id"],
                    "layer_type": row["layer_type"],
                    "activation_fn": row["activation_fn"],
                    "weight_initiations": row["weight_initiations"],
                    "input": remove_outer_quotes(row["input"]),
                    "output": remove_outer_quotes(row["output"]),
                    "dropout_rate": row["dropout_rate"],
                    "layer_fields": json.loads(row["layer_fields"]) if row["layer_fields"] else {}
                }
                layers.append(layer)

            # Construct the reference experiment entry
            reference_experiment = {
                "based_on_id": exp_id,  # Replace exp_id with based_on_id
                "loss_fn": experiment_rows[0]["loss_fn"],
                "optimization": experiment_rows[0]["optimization"],
                "normalization": experiment_rows[0]["normalization"],
                "batch_size": experiment_rows[0]["batch_size"],
                "weight_decay": experiment_rows[0]["weight_decay"],
                "learning_rate": experiment_rows[0]["learning_rate"],
                "thresh": experiment_rows[0]["thresh"],
                "layers": layers
            }

            reference_experiments.append(reference_experiment)

        print(f"Number of reference_experiments populated: {len(reference_experiments)}")  # Debug Statement

        # Step 3: Fetch options (assuming these remain unchanged)
        # Fetch loss_fn options
        cursor.execute("SELECT loss_fn FROM loss_fn")
        loss_fn_options = [row["loss_fn"] for row in cursor.fetchall()]
        print(f"Loss Function Options: {loss_fn_options}")  # Debug Statement

        # Fetch optimization options and fields
        cursor.execute("SELECT optimization, optimization_fields FROM optimization")
        optimization_fields = {
            row["optimization"]: json.loads(row["optimization_fields"]) if row["optimization_fields"] else {}
            for row in cursor.fetchall()
        }
        print(f"Optimization Fields: {optimization_fields}")  # Debug Statement

        # Fetch normalization options
        cursor.execute("SELECT normalization FROM normalization")
        normalization_options = [row["normalization"] for row in cursor.fetchall()]
        print(f"Normalization Options: {normalization_options}")  # Debug Statement

        # Fetch layer types and their fields
        cursor.execute("SELECT layer_type, layer_fields FROM layer_type")
        layer_types = {row["layer_type"]: json.loads(row["layer_fields"]) for row in cursor.fetchall()}
        print(f"Layer Types: {layer_types}")  # Debug Statement

        # Step 4: Build the JSON
        request_json = {
            "instructions": {
                "count": num,
                "focus": focus,
                "note": "In the result JSONs, replace exp_id with based_on_id."
            },
            "options": {
                "loss_fn": loss_fn_options,
                "optimization": list(optimization_fields.keys()),
                "normalization": normalization_options,
                "layer_types": layer_types,
                "optimization_fields": optimization_fields  # Explicit metadata for optimizations
            },
            "reference_experiments": reference_experiments
        }
        return json.dumps(request_json)

    except Exception as e:
        print(f"Error creating request JSON: {e}")
        raise
    finally:
        if connection:
            connection.close()


def first_gen(request_json, num, dataset_id, model):
    connection = None
    try:
        # Establish database connection
        connection = DB.get_connection()
        cursor = connection.cursor(pymysql.cursors.DictCursor)

        # Fetch dataset details
        query = """
            SELECT name, description, train_samples, test_samples, shape
            FROM processed_dataset_data
            WHERE dataset_id = %s
        """
        cursor.execute(query, (dataset_id,))
        dataset_details = cursor.fetchone()

        if not dataset_details:
            raise ValueError(f"No dataset found for dataset_id {dataset_id}")

        # Extract dataset details
        dataset_name = dataset_details["name"]
        dataset_description = dataset_details["description"]
        train_samples = dataset_details["train_samples"]
        test_samples = dataset_details["test_samples"]
        dataset_shape = dataset_details["shape"]

        # Attempt to parse shape as JSON if applicable
        try:
            parsed_shape = json.loads(dataset_shape) if isinstance(dataset_shape, str) and dataset_shape.strip().startswith('{') else dataset_shape
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing dataset_shape: {dataset_shape} - {e}")

        # Modify the focus in the request_json
        request_json["instructions"]["focus"] += (
            f" This is the first generation of tests. Please attempt to create {num} architectures for this dataset. "
            f"Dataset Details:\n"
            f"- Name: {dataset_name}\n"
            f"- Description: {dataset_description}\n"
            f"- Train Samples: {train_samples}\n"
            f"- Test Samples: {test_samples}\n"
            f"- Shape: {dataset_shape}\n"
            f"Included as a reference is a template of how the returning JSONs should look."
        )

        # Fetch optimization metadata
        cursor.execute("SELECT optimization, optimization_fields FROM optimization")
        optimizations = {
            row["optimization"]: json.loads(row["optimization_fields"]) if row["optimization_fields"]
                                                                       and row["optimization_fields"].strip() else {}
            for row in cursor.fetchall()
        }

        # Create an example JSON template with a CNN layer
        example_json = {
            "based_on_id": 0,  # First generation
            "loss_fn": "Cross Entropy Loss",
            "optimization": "Adam",
            "normalization": "StandardScaler",
            "batch_size": 32,
            "weight_decay": 0.0001,
            "learning_rate": 0.001,
            # Here's our newly included epochs field
            "epochs": 10,
            "thresh": 0.00000000000000001,
            "layers": [
                {
                    "layer_type": "Input",
                    "activation_fn": "None",
                    "weight_initiations": "None",
                    "input": parsed_shape,
                    "output": parsed_shape,
                    "dropout_rate": "None",
                    "layer_fields": {"input_shape": parsed_shape}
                },
                {
                    "layer_type": "CNN",
                    "activation_fn": "ReLU",
                    "weight_initiations": "Xavier Initialization",
                    "input": "(32, 32, 3)",
                    "output": "(30, 30, 64)",
                    "dropout_rate": "None",
                    "layer_fields": {"kernel_size": 3, "stride": 1, "padding": 0, "in_channels": 3, "out_channels": 64}
                },
                {
                    "layer_type": "Dense",
                    "activation_fn": "ReLU",
                    "weight_initiations": "Xavier Initialization",
                    "input": "(57600)",
                    "output": 128,
                    "dropout_rate": "None",
                    "layer_fields": {"units": 128}
                },
                {
                    "layer_type": "Output",
                    "activation_fn": "Softmax",
                    "weight_initiations": "Xavier Initialization",
                    "input": 128,
                    "output": 10,
                    "dropout_rate": "None",
                    "layer_fields": {"output_shape": "(10)"}
                }
            ],
            "optimization_fields": {
                "beta1": 0.9,
                "beta2": 0.999,
                "epsilon": 1e-08
            }
        }

        # Add example JSON to the reference section
        if "reference_experiments" not in request_json:
            request_json["reference_experiments"] = []
        request_json["reference_experiments"].append(example_json)

        # Send to OpenAI API
        openai_response = send_openai_request(request_json, model)
        return openai_response

    except Exception as e:
        print(f"Error in first_gen: {e}")
        raise
    finally:
        if connection:
            connection.close()


def insert_experiments_to_db(experiments, dataset_id, modification_text):
    """
    Inserts experiments into the DB according to the schema:
      - 'model' table -> one row per experiment's model
      - 'experiment' table -> one row per experiment
      - 'layer' table -> deduplicate or insert new layer
      - 'model_layer' table -> map each layer to the model in order (layer_place)

    :param experiments: Either a dictionary with key 'experiments' -> list, or directly a list of experiment dicts.
    :param dataset_id:  The ID from 'processed_dataset_data' (database_id in 'model').
    :param modification_text: Text describing why or how this experiment was generated (for experiment table).
    :return: The number of experiments successfully inserted.
    """
    connection = None
    inserted_count = 0

    try:
        # 1) Normalize the 'experiments' input.
        if isinstance(experiments, dict):
            experiments = experiments.get("experiments", [])
        if not isinstance(experiments, list):
            raise ValueError("Invalid input: 'experiments' must be a list of dictionaries.")

        # 2) Get DB connection & cursor
        connection = DB.get_connection()
        cursor = connection.cursor()

        for experiment in experiments:
            try:
                # 3) Set default values for missing fields.
                experiment.setdefault("loss_fn", "Cross Entropy Loss")
                experiment.setdefault("optimization", "Adam")
                experiment.setdefault("normalization", "StandardScaler")
                experiment.setdefault("batch_size", 32)
                experiment.setdefault("weight_decay", 0.0)
                experiment.setdefault("learning_rate", 0.001)
                experiment.setdefault("epochs", 10)  # newly included
                experiment.setdefault("optimization_fields", {})
                experiment.setdefault("layers", [])

                # 4) Insert into the 'model' table
                model_sql = """
                    INSERT INTO model (
                        database_id,
                        loss_fn,
                        optimization,
                        normalization,
                        batch_size,
                        weight_decay,
                        learning_rate,
                        thresh,
                        optimization_fields,
                        epochs
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                model_vals = (
                    dataset_id,
                    experiment["loss_fn"],
                    experiment["optimization"],
                    experiment["normalization"],
                    experiment["batch_size"],     # int
                    experiment["weight_decay"],   # float
                    experiment["learning_rate"],  # float
                    experiment.get("thresh", None),
                    json.dumps(experiment["optimization_fields"]),
                    experiment["epochs"]  # Insert epochs into model table
                )
                cursor.execute(model_sql, model_vals)
                model_id = cursor.lastrowid

                # 5) Insert into the 'experiment' table
                # Replace based_on_id=0 with None if it is 0
                based_on = experiment.get("based_on_id", 0)
                if based_on == 0:
                    based_on = None

                experiment_sql = """
                    INSERT INTO experiment (
                        based_on,
                        modification_text,
                        model_id,
                        state,
                        date,
                        sent_requests,
                        tests_done
                    )
                    VALUES (%s, %s, %s, %s, NOW(), 0, 0)
                """
                experiment_vals = (
                    based_on,
                    modification_text,
                    model_id,
                    "Waiting"
                )
                cursor.execute(experiment_sql, experiment_vals)
                exp_id = cursor.lastrowid

                # 6) For each layer, find or create a row in 'layer', then link via 'model_layer'
                layer_place = 0
                for layer in experiment["layers"]:
                    layer_fields = layer.get("layer_fields", {})
                    if isinstance(layer_fields, str):
                        try:
                            layer_fields = json.loads(layer_fields)
                        except json.JSONDecodeError:
                            layer_fields = {}

                    # Convert input/output to JSON strings
                    layer_input = json.dumps(layer.get("input")) if "input" in layer else None
                    layer_output = json.dumps(layer.get("output")) if "output" in layer else None

                    dropout_val = layer.get("dropout_rate")
                    if isinstance(dropout_val, str) and dropout_val.lower() == "none":
                        dropout_val = None
                    elif dropout_val is not None:
                        try:
                            dropout_val = float(dropout_val)
                        except ValueError:
                            dropout_val = None

                    layer_type = layer.get("layer_type", "Unknown")
                    activation_fn = layer.get("activation_fn")
                    weight_initiations = layer.get("weight_initiations")
                    layer_fields_json = json.dumps(layer_fields)

                    # optional deduplicate:
                    check_layer_sql = """
                        SELECT layer_id
                        FROM layer
                        WHERE layer_type = %s
                          AND activation_fn = %s
                          AND weight_initiations = %s
                          AND input = %s
                          AND output = %s
                          AND dropout_rate <=> %s
                          AND layer_fields = %s
                        LIMIT 1
                    """
                    cursor.execute(check_layer_sql, (
                        layer_type,
                        activation_fn,
                        weight_initiations,
                        layer_input,
                        layer_output,
                        dropout_val,
                        layer_fields_json
                    ))
                    row = cursor.fetchone()

                    if row:
                        layer_id = row[0]  # or row["layer_id"] if dict
                    else:
                        insert_layer_sql = """
                            INSERT INTO layer (
                                layer_type,
                                activation_fn,
                                weight_initiations,
                                input,
                                output,
                                dropout_rate,
                                layer_fields
                            )
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """
                        cursor.execute(insert_layer_sql, (
                            layer_type,
                            activation_fn,
                            weight_initiations,
                            layer_input,
                            layer_output,
                            dropout_val,
                            layer_fields_json
                        ))
                        layer_id = cursor.lastrowid

                    out_shape = layer_fields.get("output_shape", None)
                    if out_shape is not None:
                        out_shape = json.dumps(out_shape)
                    else:
                        out_shape = None

                    model_layer_sql = """
                        INSERT INTO model_layer (
                            model_id,
                            layer_id,
                            layer_place,
                            out_shape
                        )
                        VALUES (%s, %s, %s, %s)
                    """
                    cursor.execute(model_layer_sql, (
                        model_id,
                        layer_id,
                        layer_place,
                        out_shape
                    ))

                    layer_place += 1

                connection.commit()
                inserted_count += 1

            except Exception as e:
                if connection:
                    connection.rollback()
                print(f"Error processing experiment: {experiment}\nError: {e}")
                continue

        return inserted_count

    except Exception as e:
        if connection:
            connection.rollback()
        print(f"Error inserting experiments: {e}")
        raise

    finally:
        if connection:
            connection.close()