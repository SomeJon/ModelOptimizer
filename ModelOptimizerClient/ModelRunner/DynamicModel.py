import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import json
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer


class DynamicModel(nn.Module):
    def __init__(self, json_str):
        super(DynamicModel, self).__init__()
        # Parse the JSON string
        config = json.loads(json_str)
        experiment = config['experiment_data']

        # Store training parameters
        self.exp_id = config.get('exp_id', None)
        self.test_id = config.get('test_id', None)
        self.batch_size = experiment.get('batch_size', 32)
        self.epochs = experiment.get('epochs', 10)
        self.learning_rate = experiment.get('learning_rate', 0.001)
        self.loss_fn_name = experiment.get('loss_fn', 'CrossEntropyLoss')
        self.normalization = experiment.get('normalization', 'None')
        self.optimization = experiment.get('optimization', 'Adam')
        self.optimization_fields = experiment.get('optimization_fields', {})
        self.weight_decay = experiment.get('weight_decay', 0.0)

        # Initialize current_shape to None
        self.current_shape = None

        # Build layers
        self.layers = nn.ModuleList()
        self.build_layers(experiment['layers'])

        # Define loss function
        self.loss_fn = self.get_loss_fn(self.loss_fn_name)

        # Initialize optimizer as None; to be set during training
        self.optimizer = None

        # Handle normalization
        self.scaler = self.get_scaler(self.normalization)

    def build_layers(self, layer_configs):
        for layer in layer_configs:
            layer_type = layer['layer_type']
            activation_fn = layer.get('activation_fn', None)
            dropout_rate = layer.get('dropout_rate', None)
            layer_fields = layer.get('layer_fields', {})
            input_spec = layer.get('input')
            output_spec = layer.get('output')

            # Convert string specifications to actual values
            input_parsed = self.parse_spec(input_spec)
            output_parsed = self.parse_spec(output_spec)

            if layer_type == 'Input':
                # Store the input shape for future reference
                if isinstance(input_parsed, tuple):
                    self.current_shape = input_parsed  # (H, W, C)
                else:
                    raise ValueError(f"Invalid input specification: {input_spec}")
                continue  # Skip adding Input layer to nn.ModuleList
            elif layer_type == 'CNN':
                in_channels = layer_fields.get('in_channels')
                out_channels = layer_fields.get('out_channels')
                kernel_size = layer_fields.get('kernel_size')
                stride = layer_fields.get('stride', 1)
                padding = layer_fields.get('padding', 0)
                conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
                self.layers.append(conv)
                # Update current shape
                self.current_shape = self.compute_conv_output_shape(self.current_shape, kernel_size, stride, padding)
            elif layer_type == 'Pooling':
                pool_type = layer_fields.get('pool_type', 'max').lower().strip("'")
                pool_size = layer_fields.get('pool_size', 2)
                stride = layer_fields.get('stride', 2)
                if isinstance(pool_size, str):
                    pool_size = self.parse_spec(pool_size)
                if pool_type == 'max':
                    pool = nn.MaxPool2d(kernel_size=pool_size, stride=stride)
                elif pool_type == 'average':
                    pool = nn.AvgPool2d(kernel_size=pool_size, stride=stride)
                else:
                    raise ValueError(f"Unsupported pool type: {pool_type}")
                self.layers.append(pool)
                # Update current shape
                self.current_shape = self.compute_pool_output_shape(self.current_shape, pool_size, stride)
            elif layer_type == 'Dense':
                units = layer_fields.get('units')
                if isinstance(input_parsed, tuple):
                    # Flatten the tuple to compute input_dim
                    input_dim = 1
                    for dim in input_parsed:
                        input_dim *= dim
                else:
                    input_dim = input_parsed
                dense = nn.Linear(input_dim, units)
                self.layers.append(dense)
                self.current_shape = units  # Update current shape
            elif layer_type == 'BatchNorm':
                num_features = layer_fields.get('num_features') or layer_fields.get('out_channels') or (
                    self.current_shape[-1] if isinstance(self.current_shape, tuple) else self.current_shape)
                batch_norm = nn.BatchNorm2d(num_features)
                self.layers.append(batch_norm)
            elif layer_type == 'Dropout':
                rate = layer_fields.get('rate', 0.5)
                dropout = nn.Dropout(p=rate)
                self.layers.append(dropout)
            elif layer_type == 'Output':
                output_units = layer_fields.get('output_shape', 10)
                if isinstance(output_units, str):
                    output_units = self.parse_spec(output_units)
                # Add a Linear layer to map from current feature size to output units
                if isinstance(self.current_shape, tuple):
                    # Flatten the current shape
                    input_dim = 1
                    for dim in self.current_shape:
                        input_dim *= dim
                    linear = nn.Linear(input_dim, output_units)
                    self.layers.append(linear)
                    self.current_shape = output_units
                else:
                    # If current_shape is already an integer
                    linear = nn.Linear(self.current_shape, output_units)
                    self.layers.append(linear)
                    self.current_shape = output_units
                # Conditionally add activation function based on loss function
                if self.loss_fn_name != 'Cross Entropy Loss' and activation_fn and activation_fn != 'None':
                    activation = self.get_activation_fn(activation_fn)
                    if activation:
                        self.layers.append(activation)
                # For Cross Entropy Loss, typically no activation is added here
            else:
                raise ValueError(f"Unsupported layer type: {layer_type}")

            # Add activation function if specified and not 'None', excluding Output layer
            if layer_type != 'Output':
                if activation_fn and activation_fn != 'None':
                    activation = self.get_activation_fn(activation_fn)
                    if activation:
                        self.layers.append(activation)

            # Add dropout if specified
            if dropout_rate:
                dropout = nn.Dropout(p=dropout_rate)
                self.layers.append(dropout)

    def compute_conv_output_shape(self, input_shape, kernel_size, stride, padding, dilation=1):
        """
        Computes the output shape after a Conv2d layer.
        input_shape: tuple (H, W, C)
        Returns: tuple (H_out, W_out, C_out)
        """
        H_in, W_in, C_in = input_shape
        H_out = int((H_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
        W_out = int((W_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
        # Update C_out based on the Conv2d layer's out_channels
        conv_layer = self.layers[-1]
        if isinstance(conv_layer, nn.Conv2d):
            C_out = conv_layer.out_channels
        else:
            raise ValueError("Last layer is not Conv2d")
        return (H_out, W_out, C_out)

    def compute_pool_output_shape(self, input_shape, pool_size, stride, padding=0, dilation=1):
        """
        Computes the output shape after a Pooling layer.
        input_shape: tuple (H, W, C)
        Returns: tuple (H_out, W_out, C)
        """
        H_in, W_in, C = input_shape
        if isinstance(pool_size, tuple):
            pool_H, pool_W = pool_size
        else:
            pool_H = pool_W = pool_size
        H_out = int((H_in + 2 * padding - dilation * (pool_H - 1) - 1) / stride + 1)
        W_out = int((W_in + 2 * padding - dilation * (pool_W - 1) - 1) / stride + 1)
        return (H_out, W_out, C)

    def parse_spec(self, spec):
        """
        Parses a specification string like "(32,32,3)" or "15*15*32" into a tuple or int.
        For single-element tuples like "(10)", returns an integer 10 instead of (10,).
        """
        if isinstance(spec, str):
            spec = spec.strip()
            if spec.startswith('(') and spec.endswith(')'):
                spec_content = spec[1:-1].strip()
                if ',' in spec_content:
                    # Tuple format e.g., "(32,32,3)"
                    parts = spec_content.split(',')
                    return tuple(int(p.strip()) for p in parts)
                elif '*' in spec_content:
                    # Mathematical expression e.g., "15*15*32"
                    try:
                        return int(eval(spec_content))
                    except Exception as e:
                        raise ValueError(f"Error evaluating expression '{spec_content}': {e}")
                else:
                    # Single integer within parentheses e.g., "(10)"
                    try:
                        return int(spec_content)
                    except ValueError:
                        raise ValueError(f"Unable to parse spec: {spec}")
            else:
                # Non-parentheses string, attempt to parse as integer or expression
                if '*' in spec:
                    try:
                        return int(eval(spec))
                    except Exception as e:
                        raise ValueError(f"Error evaluating expression '{spec}': {e}")
                else:
                    try:
                        return int(spec)
                    except ValueError:
                        raise ValueError(f"Unable to parse spec: {spec}")
        else:
            # If spec is already an integer or tuple, return as is
            return spec

    def get_activation_fn(self, name):
        activations = {
            'ReLU': nn.ReLU(),
            'Tanh': nn.Tanh(),
            'Sigmoid': nn.Sigmoid(),
            'Softmax': nn.Softmax(dim=1),
            # Add more activations as needed
        }
        activation = activations.get(name, None)
        if activation is None and name != 'None':
            raise ValueError(f"Unsupported activation function: {name}")
        return activation

    def get_loss_fn(self, name):
        loss_functions = {
            'Cross Entropy Loss': nn.CrossEntropyLoss(),
            'Binary Cross Entropy': nn.BCELoss(),
            'Mean Squared Error': nn.MSELoss(),
            # Add more loss functions as needed
        }
        loss_fn = loss_functions.get(name, None)
        if loss_fn is None:
            raise ValueError(f"Unsupported loss function: {name}")
        return loss_fn

    def get_optimizer(self, parameters):
        optimizers = {
            'Adam': optim.Adam,
            'SGD': optim.SGD,
            # Add more optimizers as needed
        }
        optimizer_cls = optimizers.get(self.optimization, None)
        if optimizer_cls is None:
            raise ValueError(f"Unsupported optimizer: {self.optimization}")

        # Map JSON optimizer fields to PyTorch optimizer arguments
        optimizer_params = {}
        if self.optimization == 'Adam':
            # Extract beta1, beta2, epsilon from optimization_fields
            beta1 = self.optimization_fields.get('beta1', 0.9)
            beta2 = self.optimization_fields.get('beta2', 0.999)
            epsilon = self.optimization_fields.get('epsilon', 1e-08)
            optimizer_params['betas'] = (beta1, beta2)
            optimizer_params['eps'] = epsilon
        elif self.optimization == 'SGD':
            # SGD might have momentum, nesterov, etc.
            momentum = self.optimization_fields.get('momentum', 0.0)
            nesterov = self.optimization_fields.get('nesterov', False)
            optimizer_params['momentum'] = momentum
            optimizer_params['nesterov'] = nesterov
        # Add more optimizer-specific mappings here as needed

        # Add common parameters
        optimizer_params['lr'] = self.learning_rate
        optimizer_params['weight_decay'] = self.weight_decay

        return optimizer_cls(parameters, **optimizer_params)

    def get_scaler(self, name):
        scalers = {
            'StandardScaler': StandardScaler(),
            'MinMaxScaler': MinMaxScaler(),
            'Normalizer': Normalizer(),
            'None': None
        }
        scaler = scalers.get(name, None)
        if scaler is None and name != 'None':
            raise ValueError(f"Unsupported normalization: {name}")
        return scaler

    def forward(self, x):
        for layer in self.layers:
            # If the layer is a Linear layer and x is multi-dimensional, flatten it
            if isinstance(layer, nn.Linear):
                if x.dim() > 2:
                    x = x.view(x.size(0), -1)
            x = layer(x)
        return x

    def initialize_optimizer(self):
        """
        Initializes the optimizer. Should be called after model initialization.
        """
        self.optimizer = self.get_optimizer(self.parameters())

    def preprocess(self, x):
        """
        Applies normalization if specified.
        """
        if self.scaler:
            # Assuming x is a NumPy array; convert to tensor after scaling
            x = self.scaler.transform(x)
            x = torch.tensor(x, dtype=torch.float32)
        return x


def get_DynamicModel(json_str):
    ret_model = DynamicModel(json_str)
    ret_model.initialize_optimizer()

    return ret_model


# Example Usage
if __name__ == "__main__":
    # Example JSON string (use one from your list)
    json_str = """
    {"exp_id": 48, "experiment_data": {"based_on_id": 0, "batch_size": 128, "epochs": 10, "exp_id": 48, "layers": [{"activation_fn": "None", "dropout_rate": null, "input": "(32,32,3)", "layer_fields": {"input_shape": "(32,32,3)"}, "layer_type": "Input", "output": "(32,32,3)", "weight_initiations": null}, {"activation_fn": "Tanh", "dropout_rate": null, "input": "(32,32,3)", "layer_fields": {"in_channels": 3, "kernel_size": 5, "out_channels": 64, "padding": 0, "stride": 1}, "layer_type": "CNN", "output": "(28,28,64)", "weight_initiations": null}, {"activation_fn": "None", "dropout_rate": null, "input": "(28,28,64)", "layer_fields": {"pool_size": "(2,2)", "pool_type": "'average'", "stride": 2}, "layer_type": "Pooling", "output": "(14,14,64)", "weight_initiations": null}, {"activation_fn": "Tanh", "dropout_rate": null, "input": "(14*14*64)", "layer_fields": {"units": 120}, "layer_type": "Dense", "output": 120, "weight_initiations": null}, {"activation_fn": "Softmax", "dropout_rate": null, "input": 120, "layer_fields": {"output_shape": "(10)"}, "layer_type": "Output", "output": 10, "weight_initiations": null}], "learning_rate": 0.1, "loss_fn": "Cross Entropy Loss", "normalization": "MinMaxScaler", "optimization": "SGD", "optimization_fields": {"momentum": 0.99}, "weight_decay": 0.0}, "test_id": 60}
    """

    # Initialize the model
    model = DynamicModel(json_str)

    # Initialize optimizer
    model.initialize_optimizer()

    # Print the model architecture
    print(model)

    # Example forward pass with random input
    # Assuming input shape is (batch_size, channels, height, width)
    x = torch.randn(128, 3, 32, 32)  # Example input
    output = model(x)
    print(output.shape)  # Should match the Output layer's expected shape
