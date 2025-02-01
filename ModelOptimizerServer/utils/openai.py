import json
from utils.CustomChatCompletion import CustomChatCompletion


def send_openai_request(request_json, model, focus, num):
    """
    Sends requests to OpenAI API for each reference experiment in the input JSON,
    asking the model to generate new compact-format machine learning experiments.

    :param num: Number of experiments to generate per reference.
    :param focus: The main focus of the new experiments.
    :param request_json: JSON structure (or JSON string) with the request details.
    :param model: The model to use (e.g., "gpt-3.5-turbo", "gpt-4").
    :return: A dictionary containing a list of parsed experiments under "experiments".
    """
    try:
        # Ensure the request is a Python dictionary.
        if isinstance(request_json, str):
            request_json = json.loads(request_json)

        # Verify that reference_experiments is provided.
        if "reference_experiments" not in request_json or not request_json["reference_experiments"]:
            raise ValueError("No reference_experiments found in the input JSON.")

        # Number of experiments to generate per reference.
        num_tests = request_json.get("num_tests", 1)

        # Copy other request fields except for reference_experiments.
        base_request_data = {k: v for k, v in request_json.items() if k != "reference_experiments"}

        all_experiments = []

        # Process each reference experiment.
        for ref_exp in request_json["reference_experiments"]:
            try:
                compact_ref_exp = json_to_compact(ref_exp)
            except ValueError as e:
                print(f"Error compacting reference experiment:\n{ref_exp}\nSkipping. Error: {e}")
                continue

            base_request_data["reference_experiments"] = [compact_ref_exp]

            # Generate num_tests new experiments for this reference.
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
                                    "You are an assistant specialized in generating machine learning experiments "
                                    "in a single-line compact format.\n\n"
                                    "**Primary Objective:**\n"
                                    f"- **Focus:** {focus}.\n\n"
                                    "**Format Rules:**\n"
                                    "1. Top-level fields are separated by semicolons (`;`) in the following fixed order:\n"
                                    "   based_on_id, loss_fn, optimization, normalization, batch_size, weight_decay, "
                                    "learning_rate, epochs, min_delta, patience, layers, optimization_fields.\n"
                                    "2. Each field is formatted as key:value (with no extra spaces).\n"
                                    "3. The `layers` field consists of one or more layers separated by a pipe character (`|`).\n"
                                    "   Each layer is a comma-separated list of key=value pairs.\n"
                                    "4. For layers that support dropout, include the parameter `dropout_rate` within the layer's key=value pairs.\n"
                                    "5. Nested fields (such as `layer_fields`) should be represented as plus-separated key=value pairs.\n"
                                    "6. The `optimization_fields` should also be formatted as plus-separated key=value pairs.\n"
                                    "7. Every layer must include both `input` and `output` fields.\n"
                                    "8. Do not include any extra text or JSON outside the specified format. "
                                    "If multiple experiments are generated, each should be on a separate line.\n\n"
                                    "**Example (Multi-Layer Model with Dropout via dropout_rate):**\n"
                                    "```\n"
                                    "based_on_id:0;loss_fn:Cross Entropy Loss;optimization:Adam;min_delta:0.001;patience:2;"
                                    "normalization:StandardScaler;batch_size:32;weight_decay:0.0001;learning_rate:0.001;"
                                    "epochs:10;layers:"
                                    "activation_fn=None,input=(32,32,3),dropout_rate=0.0,layer_fields=input_shape=(32,32,3),"
                                    "layer_type=Input,output=(32,32,3)|"
                                    "activation_fn=ReLU,input=(32,32,3),dropout_rate=0.2,layer_fields=in_channels=3+kernel_size=3+"
                                    "out_channels=64+padding=1+stride=1,layer_type=CNN,weight_initiations=Xavier Initialization,"
                                    "output=(30,30,64)|"
                                    "activation_fn=ReLU,input=(30,30,64),dropout_rate=0.3,layer_fields=in_channels=64+kernel_size=3+"
                                    "out_channels=128+padding=1+stride=1,layer_type=CNN,weight_initiations=Xavier Initialization,"
                                    "output=(28,28,128)|"
                                    "activation_fn=None,input=(28,28,128),dropout_rate=0.0,layer_fields=pool_type=max+pool_size=2+"
                                    "stride=2,layer_type=Pooling,output=(14,14,128)|"
                                    "activation_fn=ReLU,input=25088,dropout_rate=0.4,layer_fields=units=256,layer_type=Dense,"
                                    "weight_initiations=Xavier Initialization,output=256|"
                                    "activation_fn=Softmax,input=256,dropout_rate=0.0,layer_fields=output_shape=10,layer_type=Output,"
                                    "output=10;optimization_fields:beta1=0.9+beta2=0.999+epsilon=1e-08\n"
                                    "```\n\n"
                                    "**Notes:**\n"
                                    "- In the `Input` layer, both `input` and `output` must be set to the input shape (e.g., `(32,32,3)`).\n"
                                    "- In the `Output` layer, `output_shape` should be set to the number of classes (e.g., `10` for CIFAR-10).\n"
                                    "- Use the `dropout_rate` parameter inside layers (when applicable) to specify dropout.\n"
                                    "- Ensure that no fields are left as `null`."
                                )
                            },
                            {
                                "role": "user",
                                "content": (
                                    f"Generate {num} new machine learning experiments in the above format.\n\n"
                                    "**Primary Objective:**\n"
                                    f"- **Focus:** {focus}.\n\n"
                                    "**Secondary Instructions:**\n"
                                    "- Attempt to generate new tests based on the `reference_experiments` with small changes.\n"
                                    "- All layers must include `input` and `output` fields.\n"
                                    "- Use `based_on_id:0` for brand-new experiments or `based_on_id:X` when referencing an existing ID.\n"
                                    "- Do not include any extra text or JSON.\n\n"
                                    f"**Request:**\n{compact_request_json}"
                                )
                            }
                        ]
                    )
                    print("Request sent.")

                    response_text = response['choices'][0]['message']['content']
                    print("OpenAI Response:\n", response_text)

                    # Process each non-empty line from the response.
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
    Convert a JSON-like dictionary into a compact string format.
    Top-level fields are output in a fixed order, and layers are represented
    as a pipe-separated list of comma-separated key=value pairs.
    Nested fields (e.g., layer_fields) are represented as plus-separated key=value pairs.
    """
    output_parts = []
    # Fixed order for top-level fields.
    top_fields_order = [
        "based_on_id", "loss_fn", "optimization", "normalization", "batch_size",
        "weight_decay", "learning_rate", "epochs", "min_delta", "patience"
    ]
    for key in top_fields_order:
        if key in data:
            output_parts.append(f"{key}:{data[key]}")

    # Process layers.
    if "layers" in data and isinstance(data["layers"], list):
        layer_strs = []
        for layer in data["layers"]:
            # Sort the layer keys for uniformity.
            items = []
            for k in sorted(layer.keys()):
                v = layer[k]
                if k == "layer_fields" and isinstance(v, dict):
                    # Sort nested keys for consistency.
                    subitems = [f"{subk}={v[subk]}" for subk in sorted(v.keys())]
                    items.append(f"{k}={'+'.join(subitems)}")
                else:
                    items.append(f"{k}={v}")
            layer_strs.append(",".join(items))
        output_parts.append(f"layers:{'|'.join(layer_strs)}")

    # Process optimization_fields.
    if "optimization_fields" in data and isinstance(data["optimization_fields"], dict):
        opt_items = [f"{k}={data['optimization_fields'][k]}" for k in sorted(data["optimization_fields"].keys())]
        output_parts.append(f"optimization_fields:{'+'.join(opt_items)}")

    return ";".join(output_parts)


def compact_to_json(compact_str: str) -> dict:
    """
    Parse the compact string format back into a Python dictionary.
    Expected format: top-level fields separated by semicolons, with layers and
    optimization_fields encoded as specified.
    """
    def parse_value(s: str):
        s = s.strip()
        if s.lower() == "none":
            return None
        try:
            return int(s)
        except ValueError:
            pass
        try:
            return float(s)
        except ValueError:
            pass
        return s

    def split_on_commas_outside_parens(line: str):
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
        if current:
            result.append("".join(current).strip())
        return result

    def remove_code_block_markers(s: str) -> str:
        return s.replace('```', '').strip()

    compact_str = remove_code_block_markers(compact_str)
    result = {}
    top_fields = [fld.strip() for fld in compact_str.split(";") if fld.strip()]
    for fld in top_fields:
        if ":" not in fld:
            continue
        key, val = fld.split(":", 1)
        key = key.strip()
        val = val.strip()
        if key == "layers":
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
                        lf_dict = {}
                        if sub_val:
                            for pair in sub_val.split("+"):
                                if "=" in pair:
                                    pk, pv = pair.split("=", 1)
                                    lf_dict[pk.strip()] = parse_value(pv.strip())
                        layer_dict["layer_fields"] = lf_dict
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
            result[key] = parse_value(val)

    if not result:
        raise ValueError("Parsed result is empty. Please check the input format.")
    return result
