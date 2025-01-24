import json
from utils.CustomChatCompletion import CustomChatCompletion


def send_openai_request(request_json, model):
    """
    Sends requests to OpenAI API for each reference experiment in the input JSON.

    :param request_json: JSON structure for OpenAI API, either as a dictionary or JSON string.
    :return: Parsed JSON responses from OpenAI API.
    """
    try:
        # Ensure request_json is a dictionary
        if isinstance(request_json, str):
            request_json = json.loads(request_json)

        # Check if "reference_experiments" exists and is not empty
        if "reference_experiments" not in request_json or not request_json["reference_experiments"]:
            raise ValueError("No reference_experiments found in the input JSON.")

        # Extract other fields from the original request
        num_tests = request_json.get("num_tests", 1)
        base_request_data = {
            k: v for k, v in request_json.items() if k != "reference_experiments"
        }

        # Accumulate results
        all_experiments = []

        for ref_exp in request_json["reference_experiments"]:
            try:
                compact_ref_exp = json_to_compact(ref_exp)
            except ValueError as e:
                print(f"Error compacting reference experiment: {ref_exp}. Skipping. Error: {e}")
                continue  # Skip this reference experiment

            base_request_data["reference_experiments"] = [compact_ref_exp]

            # Generate `num_tests` experiments for this reference
            for _ in range(num_tests):
                compact_request_json = json.dumps(base_request_data, indent=2)

                try:
                    print("Sending request to OpenAI...")
                    response = CustomChatCompletion.create(
                        model=model,
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You generate machine learning experiments. Compact format rules:"
                                    " Follow these compact format rules strictly:"
                                    "- Key-value pairs separated by ';'."
                                    "- Replace 'exp_id' with 'based_on_id', and input null or the id of original."
                                    "- 'layers': layer_type{field1=value1+field2=value2}, separated by '|'."
                                    "- 'optimization_fields': field1=value1+field2=value2."
                                    "- No extra text, comments, or formatting."
                                    "\n\n"
                                    "For example:\n"
                                    "based_on_id:1;loss_fn:Cross Entropy Loss;optimization:Adam;normalization:StandardScaler;"
                                    "batch_size:32;weight_decay:0.0001;learning_rate:0.001;"
                                    "layers:Input{input_shape='(32, 32, 3)'}|Dense{units=128}|Output{output_shape=[10]};"
                                    "optimization_fields:beta1=0.9+beta2=0.999+epsilon=1e-8\n"
                                )
                            },
                            {
                                "role": "user",
                                "content": (
                                    "The following is a compact format request to generate machine learning experiments.\n\n"
                                    "Instructions:\n"
                                    "- Return the response in the same compact format as described above.\n"
                                    "- Do not include extra text, explanations, or formatting.\n"
                                    "- Ensure the response strictly follows the given structure.\n\n"
                                    f"Request:\n{compact_request_json}"
                                )
                            }
                        ]
                    )
                    print("Request sent.")

                    # Parse the response
                    result = response['choices'][0]['message']['content']

                    # Validate and parse the response
                    compact_response = result.strip()
                    for compact_exp in compact_response.split("\n"):
                        if compact_exp.strip():  # Skip empty lines
                            try:
                                all_experiments.append(compact_to_json(compact_exp))
                            except ValueError as e:
                                print(f"Error parsing response: {compact_exp}. Skipping. Error: {e}")
                                continue  # Skip this experiment
                except Exception as e:
                    print(f"Error during OpenAI request for reference experiment {ref_exp}. Skipping. Error: {e}")
                    continue  # Skip this reference experiment

        return {"experiments": all_experiments}

    except Exception as e:
        print(f"Error during OpenAI request: {e}")
        raise


def compact_to_json(compact_str):
    """
    Converts a compact string representation of an experiment into a JSON object.
    """
    try:
        parts = compact_str.split(";")
        experiment = {}

        for part in parts:
            if ":" not in part and "=" not in part:  # Accept both ':' and '='
                raise ValueError(f"Malformed part in compact string: {part}")

            # Check if it's a key-value pair using ':' or '='
            delimiter = ":" if ":" in part else "="
            key, value = part.split(delimiter, 1)

            if key == "layers":
                layers = []
                for layer in value.split("|"):
                    if "{" not in layer or "}" not in layer:
                        raise ValueError(f"Malformed layer definition: {layer}")
                    layer_type, fields = layer.split("{", 1)
                    fields = fields.rstrip("}")
                    layer_data = {"layer_type": layer_type.strip()}
                    for field in fields.split("+"):
                        if "=" not in field:
                            raise ValueError(f"Malformed field in layer: {field}")
                        k, v = field.split("=", 1)
                        v = v.strip().strip("'").strip('"')
                        if v.replace('.', '', 1).isdigit():
                            v = float(v) if '.' in v else int(v)
                        elif v.startswith("(") and v.endswith(")"):
                            v = tuple(map(int, v.strip("()").split(",")))
                        elif v.startswith("[") and v.endswith("]"):
                            v = json.loads(v)
                        layer_data[k.strip()] = v
                    layers.append(layer_data)
                experiment["layers"] = layers
            elif key == "optimization_fields":
                fields = value.split("+")
                experiment[key] = {
                    k.strip(): float(v.strip())
                    for field in fields
                    for k, v in [field.split("=", 1)]
                }
            else:
                value = value.strip().strip("'").strip('"')
                if value == "null":
                    experiment[key] = None
                elif value.replace('.', '', 1).isdigit():
                    experiment[key] = float(value) if '.' in value else int(value)
                else:
                    experiment[key] = value

        return experiment
    except Exception as e:
        raise ValueError(f"Error in compact_to_json: {e}, Input: {compact_str}")


def json_to_compact(json_data):
    """
    Converts a JSON-compatible dictionary into a compact string format.
    """
    try:
        compact_str = []

        # Add top-level fields
        for key, value in json_data.items():
            if key == "layers":
                layers = []
                for layer in value:
                    layer_type = layer["layer_type"]
                    fields = "+".join([f"{k}={repr(v)}" for k, v in layer.items() if k != "layer_type"])
                    layers.append(f"{layer_type}{{{fields}}}")
                compact_str.append(f"layers:{'|'.join(layers)}")
            elif key == "optimization_fields":
                fields = "+".join([f"{k}={v}" for k, v in value.items()])
                compact_str.append(f"{key}:{fields}")
            else:
                compact_str.append(f"{key}:{value}")

        return ";".join(compact_str)

    except Exception as e:
        raise ValueError(f"Error in json_to_compact: {e}, Input: {json_data}")
