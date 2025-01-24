import json


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



# Example compact response with updated format
response = """
based_on_id:null;loss_fn:Binary Cross Entropy;optimization:SGD;normalization:None;batch_size:32;weight_decay=0.0002;learning_rate:0.002;layers:Input{input_shape='(32, 32, 3)'}|CNN{stride=2+padding=2+in_channels=3+kernel_size=5+out_channels=32}|BatchNorm{epsilon=1e-6+momentum=0.98}|Dropout{rate=0.25}|CNN{stride=1+padding=1+in_channels=32+kernel_size=3+out_channels=64}|Pooling{type='average'+stride=2+pool_size=(2,2)}|BatchNorm{epsilon=1e-6+momentum=0.98}|Dropout{rate=0.35}|Dense{units=128}|BatchNorm{epsilon=1e-6+momentum=0.98}|Dropout{rate=0.45}|Dense{units=256}|BatchNorm{epsilon=1e-6+momentum=0.98}|Output{output_shape=[10]};optimization_fields:momentum=0.8
based_on_id:0;loss_fn:Mean Squared Error;optimization:SGD;normalization:Normalizer;batch_size:32;weight_decay:0.0002;learning_rate:0.01;layers:Input{input_shape=(32, 32, 3)}|CNN{stride=2+padding=0+in_channels=3+kernel_size=5+out_channels=16}|Dropout{rate=0.4}|Dense{units=64}|Pooling{type=average+stride=2+pool_size=2}|Output{output_shape=[10]};optimization_fields:momentum=0.8
based_on_id:0;loss_fn:Binary Cross Entropy;optimization:Adam;normalization:MinMaxScaler;batch_size:128;weight_decay:0.0005;learning_rate:0.002;layers:Input{input_shape=(32, 32, 3)}|CNN{stride=1+padding=1+in_channels=3+kernel_size=5+out_channels=64}|BatchNorm{epsilon=1e-5+momentum=0.2}|Pooling{type=max+stride=2+pool_size=2}|Dense{units=256}|Output{output_shape=[10]};optimization_fields:beta1=0.9+beta2=0.999+epsilon=1e-8
"""

# Process each line of the response
parsed_experiments = []
for line in response.strip().split("\n"):
    try:
        parsed_experiment = compact_to_json(line.strip())
        parsed_experiments.append(parsed_experiment)
        print(json.dumps(parsed_experiment, indent=2))
    except ValueError as e:
        print(f"Error: {e}")

# Optionally, save parsed experiments to a JSON file
with open("parsed_experiments.json", "w") as f:
    json.dump(parsed_experiments, f, indent=2)
