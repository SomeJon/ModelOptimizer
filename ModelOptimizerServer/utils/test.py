import json

def json_to_compact(data: dict) -> str:
    """
    Convert a JSON-like dictionary into a compact string format:
      based_on_id:...,loss_fn:...,optimization:...,...
      layers: layer1|layer2|...
      optimization_fields: key1=val1+key2=val2+...
    Each layer has comma-separated fields, including "layer_fields" plus-separated dict.
    """

    output_parts = []

    # 1. Top-level fields (use the order you want):
    top_level_order = [
        "based_on_id",
        "loss_fn",
        "optimization",
        "normalization",
        "batch_size",
        "weight_decay",
        "learning_rate"
    ]

    # Append these fields if present
    for key in top_level_order:
        if key in data:
            # Convert the value to a string
            output_parts.append(f"{key}:{data[key]}")

    # 2. layers
    if "layers" in data and isinstance(data["layers"], list):
        layer_strs = []
        for layer in data["layers"]:
            layer_items = []
            for k, v in layer.items():
                if k == "layer_fields" and isinstance(v, dict):
                    # plus-separated
                    plus_list = []
                    # We won't parse or transform sub-values, everything as string
                    for subk, subv in v.items():
                        plus_list.append(f"{subk}={subv}")
                    layer_items.append(f"layer_fields={'+'.join(plus_list)}")
                else:
                    # Everything else is key=value, as string
                    layer_items.append(f"{k}={v}")
            # comma-join for each layer
            layer_strs.append(",".join(layer_items))

        # Join multiple layers with '|'
        layers_compact_str = "|".join(layer_strs)
        output_parts.append(f"layers:{layers_compact_str}")

    # 3. optimization_fields
    if "optimization_fields" in data and isinstance(data["optimization_fields"], dict):
        opt_items = []
        for k, v in data["optimization_fields"].items():
            opt_items.append(f"{k}={v}")
        opt_compact_str = "+".join(opt_items)
        output_parts.append(f"optimization_fields:{opt_compact_str}")

    # 4. Join everything with ';'
    return ";".join(output_parts)


def compact_to_json(compact_str: str) -> dict:
    """
    Parse the compact string format back into a Python dict, without
    splitting on commas inside parentheses.

    Example:
      based_on_id:0;loss_fn:Cross Entropy;...;layers:layer_type=Input,input=(32,32,3),...
    """
    def parse_value(s: str):
        s = s.strip()
        # Optionally turn "None" into actual None:
        if s == "None":
            return "None"  # or return None
        # Try numeric:
        try:
            return int(s)
        except ValueError:
            pass
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
                # We split here
                result.append("".join(current).strip())
                current = []
            else:
                current.append(char)

        # Append any leftover piece
        if current:
            result.append("".join(current).strip())

        return result

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
            # e.g. "layer_type=Input,input=(32,32,3)...|layer_type=CNN,input=(32,32,3)..."
            layer_specs = val.split("|")
            layers_list = []

            for layer_str in layer_specs:
                layer_dict = {}
                # Instead of splitting directly on ',', we do:
                sub_fields = split_on_commas_outside_parens(layer_str)

                for sf in sub_fields:
                    if "=" not in sf:
                        continue
                    sub_key, sub_val = sf.split("=", 1)
                    sub_key = sub_key.strip()
                    sub_val = sub_val.strip()

                    if sub_key == "layer_fields":
                        # plus-separated
                        lf_dict = {}
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
            # plus-separated
            opt_dict = {}
            for pair in val.split("+"):
                if "=" in pair:
                    pk, pv = pair.split("=", 1)
                    opt_dict[pk.strip()] = parse_value(pv.strip())
            result[key] = opt_dict

        else:
            # normal top-level
            result[key] = parse_value(val)

    return result



# -------------------------------------------------------------------
# Example usage with your sample dictionary:
if __name__ == "__main__":
    example_dict = {
        "based_on_id": 0,  # Indicates the first generation
        "loss_fn": "Cross Entropy Loss",
        "optimization": "Adam",
        "normalization": "StandardScaler",
        "batch_size": 32,
        "weight_decay": 0.0001,
        "learning_rate": 0.001,
        "layers": [
            {
                "layer_type": "Input",
                "activation_fn": "None",
                "weight_initiations": "None",
                "input": "(32, 32, 3)",
                "output": "(32, 32, 3)",
                "dropout_rate": "None",
                "layer_fields": {"input_shape": "(32, 32, 3)"}
            },
            {
                "layer_type": "CNN",
                "activation_fn": "ReLU",
                "weight_initiations": "Xavier Initialization",
                "input": "(32, 32, 3)",
                "output": "(30, 30, 64)",
                "dropout_rate": "None",
                "layer_fields": {
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 0,
                    "in_channels": 3,
                    "out_channels": 64
                }
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

    print("=== Original Dictionary ===")
    print(json.dumps(example_dict, indent=2))

    # Convert to compact string
    compact_string = json_to_compact(example_dict)
    print("\n=== Compact String ===")
    print(compact_string)

    # Convert back to dict
    roundtrip_dict = compact_to_json(compact_string)

    print("\n=== Round-Tripped Dictionary ===")
    print(json.dumps(roundtrip_dict, indent=2))

    # If you want to check they are "the same" ignoring key order:
    # Compare as Python objects:
    # (But note: original numeric values are now strings, etc.)
    # so you'll see differences like "0.0001" vs 0.0001.
    # If that's acceptable, then the structure + keys + values
    # are effectively the same in string form.
