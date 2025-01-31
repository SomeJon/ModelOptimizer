import json
from utils.CustomChatCompletion import CustomChatCompletion


def send_openai_request(request_json, model, focus, num):
    """
    Sends requests to OpenAI API for each reference experiment in the input JSON,
    asking the model to generate new compact-format machine learning experiments.

    :param num: number to generate
    :param focus: main focus
    :param request_json: JSON structure or JSON string for the OpenAI request.
    :param model: The model to use (e.g., "gpt-3.5-turbo" or "gpt-4").
    :return: A dictionary containing a list of parsed experiments under "experiments".
    """
    try:
        # Ensure Python dictionary
        if isinstance(request_json, str):
            request_json = json.loads(request_json)

        # Verify reference_experiments
        if "reference_experiments" not in request_json or not request_json["reference_experiments"]:
            raise ValueError("No reference_experiments found in the input JSON.")

        # How many experiments to generate for each reference
        num_tests = request_json.get("num_tests", 1)

        # Copy any other request fields
        base_request_data = {
            k: v for k, v in request_json.items() if k != "reference_experiments"
        }

        all_experiments = []

        # Process each reference experiment
        for ref_exp in request_json["reference_experiments"]:
            try:
                # Convert the reference experiment into a single-line compact format
                compact_ref_exp = json_to_compact(ref_exp)
            except ValueError as e:
                print(f"Error compacting reference experiment:\n{ref_exp}\nSkipping. Error: {e}")
                continue

            base_request_data["reference_experiments"] = [compact_ref_exp]

            # Generate num_tests new experiments for this reference
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
                                    "You are an assistant specialized in generating machine learning experiments in a single-line compact format.\n\n"
                                    "**Primary Objective:**\n"
                                    f"- **Focus:** {focus}.\n\n"
                                    "**Format Rules:**\n"
                                    "1. **Top-level fields** are separated by semicolons (`;`):\n"
                                    "`based_on_id`, `loss_fn`, `optimization`, `normalization`, `batch_size`, "
                                    "`weight_decay`, `learning_rate`, `epochs`, `layers`, `optimization_fields`.\n"
                                    "2. **`based_on_id`:** Set to `0` if the experiment is brand-new. If derived from "
                                    "an existing experiment with ID X, set to `based_on_id:X`.\n"
                                    "3. **Early Stopping Parameters:** Include `min_delta` and `patience` related to "
                                    "early stopping.\n"
                                    "4. **`layers`:** Each layer is a pipe-separated (`|`) string of comma-separated "
                                    "key=value pairs. Nested fields (`layer_fields`) use plus-separated (`+`) "
                                    "key=value pairs within the same comma-separated section.\n"
                                    "5. **`optimization_fields`:** Plus-separated (`+`) key=value pairs, e.g., "
                                    "`optimization_fields:beta1=0.9+beta2=0.999+epsilon=1e-8`.\n"
                                    "6. **Layer Requirements:** Every layer must include both `input` and `output` "
                                    "fields.\n"
                                    "7. **`epochs`:** Must be included as a top-level field (default is `10` if not "
                                    "specified).\n"
                                    "8. **No Extra Text or JSON:** The output should strictly follow the specified "
                                    "format. If multiple experiments are generated, place each on a new line.\n\n"
                                    "**Example (4-layer CNN with Early Stopping):**\n"
                                    "```\n"
                                    "based_on_id:0;loss_fn:Cross Entropy "
                                    "Loss;optimization:Adam;min_delta:0.001;patience:2;normalization:StandardScaler;"
                                    "batch_size=32;weight_decay=0.0001;learning_rate=0.001;epochs=10;layers:"
                                    "layer_type=Input,activation_fn=None,input=(32,32,3),output=(32,32,3),"
                                    "layer_fields=input_shape=(32,32,3)|"
                                    "layer_type=CNN,activation_fn=ReLU,weight_initiations=Xavier Initialization,"
                                    "input=(32,32,3),output=(30,30,64),dropout_rate=0.3,"
                                    "layer_fields=kernel_size=3+stride=1+padding=2+in_channels=3+out_channels=64|"
                                    "layer_type=Pooling,input=(30,30,64),output=(15,15,64),pool_type=max,"
                                    "pool_size=2+stride=2|"
                                    "layer_type=Dense,activation_fn=ReLU,weight_initiations=Xavier Initialization, "
                                    "input=14400,output=128,dropout_rate=0.3,layer_fields=units=128|"
                                    "layer_type=Output,activation_fn=Softmax,input=128,output=10,"
                                    "layer_fields=output_shape=10;optimization_fields:beta1=0.9+beta2=0.999+epsilon"
                                    "=1e-08\n"
                                    "```\n\n"
                                    "**Notes:**\n"
                                    "- In the `Input` layer, `input` and `output` should both be set to the input "
                                    "shape `(32,32,3)`.\n"
                                    "- In the `Output` layer, `output_shape` should be set to the number of classes ("
                                    "e.g., `10` for CIFAR-10).\n"
                                    "- Ensure that all required fields are populated and no fields are left as `null`."
                                )
                            },
                            {
                                "role": "user",
                                "content": (
                                    f"Generate {num} new machine learning experiments in the above format.\n\n"
                                    "**Primary Objective:**\n"
                                    f"- **Focus:** {focus}.\n\n"
                                    "**Secondary Instructions:**\n"
                                    "- Unless told otherwise, attempt to generate new tests according to the "
                                    "`reference_experiments` with small changes.\n"
                                    "- All layers must have `input` and `output`.\n"
                                    "- Use `based_on_id:0` if the experiment is brand-new, or `based_on_id:X` if "
                                    "referencing an existing ID.\n"
                                    "- No extra text or JSON.\n\n"
                                    f"**Request:**\n{compact_request_json}"
                                )
                            }
                        ]
                    )
                    print("Request sent.")

                    # Extract response text
                    response_text = response['choices'][0]['message']['content']
                    print("OpenAI Response:\n", response_text)

                    # Parse line-by-line
                    for line in response_text.split("\n"):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            parsed_exp = compact_to_json(line)
                            all_experiments.append(parsed_exp)
                        except ValueError as e:
                            print(f"Error parsing response line:\n{line}\nSkipping. Error: {e}")
                            continue

                except Exception as e:
                    print(f"Error during OpenAI request for reference experiment: {ref_exp}\nError: {e}")
                    continue

        return {"experiments": all_experiments}

    except Exception as e:
        print(f"Error during OpenAI request: {e}")
        raise


def json_to_compact(data: dict) -> str:
    """
    Convert a JSON-like dictionary into the compact string format.
    - Top-level fields: semicolons (e.g. based_on_id:..., loss_fn:..., ...).
    - layers: each layer is comma-separated key=value, layers separated by '|'.
    - layer_fields: plus-separated pairs.
    - optimization_fields: plus-separated pairs.
    """
    output_parts = []

    # Ordered top-level fields (added "epochs" at the end)
    top_fields_order = [
        "based_on_id",
        "loss_fn",
        "optimization",
        "normalization",
        "batch_size",
        "weight_decay",
        "learning_rate",
        "epochs"
    ]

    # Add each top-level field if present
    for key in top_fields_order:
        if key in data:
            output_parts.append(f"{key}:{data[key]}")

    # Handle layers
    if "layers" in data and isinstance(data["layers"], list):
        layer_strs = []
        for layer in data["layers"]:
            layer_items = []
            for k, v in layer.items():
                if k == "layer_fields" and isinstance(v, dict):
                    # plus-separated for subkeys
                    plus_list = [f"{subk}={subv}" for subk, subv in v.items()]
                    layer_items.append(f"layer_fields={'+'.join(plus_list)}")
                else:
                    layer_items.append(f"{k}={v}")
            layer_strs.append(",".join(layer_items))
        output_parts.append(f"layers:{'|'.join(layer_strs)}")

    # Handle optimization_fields
    if "optimization_fields" in data and isinstance(data["optimization_fields"], dict):
        opt_items = [f"{k}={v}" for k, v in data["optimization_fields"].items()]
        output_parts.append(f"optimization_fields:{'+'.join(opt_items)}")

    return ";".join(output_parts)


def compact_to_json(compact_str: str) -> dict:
    """
    Parse the compact string format back into a Python dict,
    including the newly added 'epochs' field if present.

    Example:
      based_on_id:0;loss_fn:Cross Entropy;...;epochs:20;layers:layer_type=Input,input=(32,32,3),...
    """
    def parse_value(s: str):
        s = s.strip()
        # Optionally turn "None" into actual None:
        if s.lower() == "none":
            return None
        # Try integer:
        try:
            return int(s)
        except ValueError:
            pass
        # Try float:
        try:
            return float(s)
        except ValueError:
            pass
        # Fallback: keep string
        return s

    def split_on_commas_outside_parens(line: str):
        """
        Splits `line` by commas that are NOT inside parentheses.
        E.g., "input=(32,32,3),output=(32,32,3)" -> ["input=(32,32,3)", "output=(32,32,3)"]
        """
        result = []
        current = []
        paren_depth = 0

        for char in line:
            if char == '(':
                paren_depth += 1
                current.append(char)
            elif char == ')':
                paren_depth = max(paren_depth - 1, 0)
                current.append(char)
            elif char == ',' and paren_depth == 0:
                result.append("".join(current).strip())
                current = []
            else:
                current.append(char)

        # Append any leftover piece
        if current:
            result.append("".join(current).strip())

        return result

    def remove_code_block_markers(s: str) -> str:
        """
        Removes code block markers (e.g., ```) from the string.
        """
        return s.replace('```', '').strip()

    # Step 1: Remove code block markers if present
    compact_str = remove_code_block_markers(compact_str)

    result = {}
    # Split top-level by ';'
    top_fields = [fld.strip() for fld in compact_str.split(";") if fld.strip()]

    for fld in top_fields:
        if ":" not in fld:
            continue
        key, val = fld.split(":", 1)
        key = key.strip()
        val = val.strip()

        if key == "layers":
            # e.g. "layer_type=Input,activation_fn=None, ... | layer_type=CNN, ..."
            layer_specs = val.split("|")
            layers_list = []

            for layer_str in layer_specs:
                layer_dict = {}
                sub_fields = split_on_commas_outside_parens(layer_str)

                for sf in sub_fields:
                    if "=" not in sf:
                        continue
                    sub_key, sub_val = sf.split("=", 1)
                    sub_key = sub_key.strip()
                    sub_val = sub_val.strip()

                    if sub_key == "layer_fields":
                        if sub_val:  # Ensure there are layer_fields to parse
                            # plus-separated
                            lf_dict = {}
                            for pair in sub_val.split("+"):
                                if "=" in pair:
                                    pk, pv = pair.split("=", 1)
                                    lf_dict[pk.strip()] = parse_value(pv.strip())
                            layer_dict["layer_fields"] = lf_dict
                        else:
                            layer_dict["layer_fields"] = {}
                    else:
                        layer_dict[sub_key] = parse_value(sub_val)

                layers_list.append(layer_dict)

            result["layers"] = layers_list

        elif key == "optimization_fields":
            opt_dict = {}
            for pair in val.split("+"):
                if "=" in pair:
                    pk, pv = pair.split("=", 1)
                    opt_dict[pk.strip()] = parse_value(pv.strip())
            result[key] = opt_dict

        else:
            # normal top-level (including epochs)
            result[key] = parse_value(val)

    # Step 2: Check if the result is empty
    if not result:
        raise ValueError("Parsed result is empty. Please check the input format.")

    #print("After parsing to json:", result)
    return result

