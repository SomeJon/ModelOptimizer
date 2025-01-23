from utils.DB import DB
import pymysql
import json

from utils.openai import send_openai_request


# Format results as an aligned table-like string
def format_as_aligned_table(headers, rows):
    # Calculate column widths
    column_widths = [len(header) for header in headers]
    for row in rows:
        column_widths = [max(width, len(str(value))) for width, value in zip(column_widths, row)]

    # Format the header
    header_line = " | ".join(f"{header:<{column_widths[i]}}" for i, header in enumerate(headers))
    separator_line = "-+-".join("-" * width for width in column_widths)

    # Format the rows
    row_lines = [
        " | ".join(f"{str(value):<{column_widths[i]}}" for i, value in enumerate(row))
        for row in rows
    ]

    # Combine everything into a single string
    return f"{header_line}\n{separator_line}\n" + "\n".join(row_lines)


def create_request_json(num, dataset_id, focus, num_of_based):
    """
    Create a request JSON for generating new experiments.

    :param num: Number of tests to generate.
    :param dataset_id: The ID of the dataset to base the tests on.
    :param focus: Instructions for the focus of the tests.
    :param num_of_based: Number of top experiments to use as references.
    :return: JSON with the request structure for OpenAI.
    """
    connection = None
    try:
        # Establish DB connection
        connection = DB.get_connection()
        cursor = connection.cursor(pymysql.cursors.DictCursor)

        # Fetch top experiments
        query_experiments = """
            SELECT exp.*, m.*, ml.layer_id, l.*
            FROM experiment exp
            INNER JOIN model m ON exp.model_id = m.model_id
            INNER JOIN model_layer ml ON m.model_id = ml.model_id
            INNER JOIN layer l ON ml.layer_id = l.layer_id
            INNER JOIN test t ON exp.exp_id = t.exp_id
            WHERE exp.dataset_id = %s
            ORDER BY t.score DESC
            LIMIT %s
        """
        cursor.execute(query_experiments, (dataset_id, num_of_based))
        experiments = cursor.fetchall()

        # Group experiments and layers
        reference_experiments = {}
        for row in experiments:
            exp_id = row["exp_id"]
            if exp_id not in reference_experiments:
                reference_experiments[exp_id] = {
                    "based_on_id": exp_id,  # Replace exp_id with based_on_id
                    "loss_fn": row["loss_fn"],
                    "optimization": row["optimization"],
                    "normalization": row["normalization"],
                    "batch_size": row["batch_size"],
                    "weight_decay": row["weight_decay"],
                    "learning_rate": row["learning_rate"],
                    "layers": []
                }
            reference_experiments[exp_id]["layers"].append({
                "layer_id": row["layer_id"],
                "layer_type": row["layer_type"],
                "activation_fn": row["activation_fn"],
                "weight_initiations": row["weight_initiations"],
                "input": row["input"],
                "output": row["output"],
                "dropout_rate": row["dropout_rate"],
                "meta_data": json.loads(row["meta_data"]) if row["meta_data"] else {}
            })

        # Fetch options
        cursor.execute("SELECT loss_fn FROM loss_fn")
        loss_fn_options = [row["loss_fn"] for row in cursor.fetchall()]

        cursor.execute("SELECT optimization FROM optimization")
        optimization_options = [row["optimization"] for row in cursor.fetchall()]

        cursor.execute("SELECT normalization FROM normalization")
        normalization_options = [row["normalization"] for row in cursor.fetchall()]

        cursor.execute("SELECT layer_type, meta_data FROM layer_type")
        layer_types = {row["layer_type"]: json.loads(row["meta_data"]) for row in cursor.fetchall()}

        # Build the JSON
        request_json = {
            "instructions": {
                "count": num,
                "focus": focus,
                "note": "In the result jsons, replace exp_id with based_on_id."
            },
            "options": {
                "loss_fn": loss_fn_options,
                "optimization": optimization_options,
                "normalization": normalization_options,
                "layer_types": layer_types
            },
            "reference_experiments": list(reference_experiments.values())
        }

        return json.dumps(request_json, indent=2)

    except Exception as e:
        print(f"Error creating request JSON: {e}")
        raise
    finally:
        if connection:
            connection.close()


def first_gen(request_json, num, dataset_id):
    """
    Handles the first generation of tests when no reference experiments exist.
    Fetches dataset details and modifies the focus accordingly.

    :param request_json: The base JSON created by create_request_json.
    :param num: Number of experiments to generate.
    :param dataset_id: The ID of the dataset for which tests are generated.
    :return: The OpenAI API response.
    """
    connection = None
    try:
        # Establish database connection
        connection = DB.get_connection()
        cursor = connection.cursor(pymysql.cursors.DictCursor)

        # Fetch dataset details
        query = """
            SELECT name, description, size, shape
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
        dataset_size = dataset_details["size"]
        dataset_shape = dataset_details["shape"]

        # Modify the focus in the request_json
        request_json["instructions"]["focus"] += (
            f" This is the first generation of tests. Please attempt to create {num} architectures for this dataset. "
            f"Dataset Details:\n"
            f"- Name: {dataset_name}\n"
            f"- Description: {dataset_description}\n"
            f"- Size: {dataset_size} MB\n"
            f"- Shape: {dataset_shape}\n"
            f"Included as a reference is a template of how the returning JSONs should look."
        )

        # Create an example JSON template
        example_json = {
            "based_on_id": 0,  # Indicates the first generation
            "loss_fn": "Cross Entropy Loss",
            "optimization": "Adam",
            "normalization": "StandardScaler",
            "batch_size": 32,
            "weight_decay": 0.0001,
            "learning_rate": 0.001,
            "layers": [
                {
                    "layer_type": "Dense",
                    "activation_fn": "ReLU",
                    "weight_initiations": "Xavier Initialization",
                    "input": 128,
                    "output": 64,
                    "dropout_rate": None,
                    "meta_data": {"units": 64}
                },
                {
                    "layer_type": "Output",
                    "activation_fn": "Softmax",
                    "weight_initiations": None,
                    "input": 64,
                    "output": 10,
                    "dropout_rate": None,
                    "meta_data": {"output_shape": [10]}
                }
            ]
        }

        # Add example JSON to the reference section
        request_json["reference_experiments"].append(example_json)

        # Send to OpenAI API
        openai_response = send_openai_request(request_json)

        return openai_response

    except Exception as e:
        print(f"Error in first_gen: {e}")
        raise
    finally:
        if connection:
            connection.close()
