import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
import json
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer

torch.manual_seed(0)


def sanitize_string(value, to_lower=True):
    """
    Sanitizes a string by stripping surrounding quotes and whitespace.

    Args:
        value (str): The string to sanitize.
        to_lower (bool): Whether to convert the string to lowercase for uniformity.

    Returns:
        str: The sanitized string, or 'none' if the input is None.
    """
    if isinstance(value, str):
        sanitized = value.strip().strip('\'"')
        return sanitized.lower() if to_lower else sanitized
    elif value is None:
        return 'none'
    return value


class DynamicModel(nn.Module):
    def __init__(self, json_input):
        super(DynamicModel, self).__init__()

        # Determine if json_input is a string or a dictionary
        if isinstance(json_input, str):
            try:
                config = json.loads(json_input)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON string provided: {e}")
        elif isinstance(json_input, dict):
            config = json_input
        else:
            raise TypeError("json_input must be a JSON string or dictionary.")

        # Ensure 'experiment_data' is present
        if 'experiment_data' not in config:
            raise KeyError("'experiment_data' field is missing in the JSON configuration.")

        experiment = config['experiment_data']

        # Store training parameters
        self.exp_id = config.get('exp_id', None)
        self.test_id = config.get('test_id', None)
        self.batch_size = experiment.get('batch_size', 32)
        self.epochs = experiment.get('epochs', 10)
        self.learning_rate = experiment.get('learning_rate', 0.001)

        # Sanitize loss function
        self.loss_fn_name = sanitize_string(experiment.get('loss_fn', 'Cross Entropy Loss')).title()

        # Sanitize optimization
        self.optimization = sanitize_string(experiment.get('optimization', 'Adam')).title()

        self.optimization_fields = experiment.get('optimization_fields', {})
        self.weight_decay = experiment.get('weight_decay', 0.0)

        # Initialize current_shape to None
        self.current_shape = None  # Will be a tuple (H, W, C) for Conv layers or int for Dense layers

        # Define number of classes (assuming CIFAR-10; adjust as needed)
        self.num_classes = 10  # You can make this dynamic based on your dataset

        # Initialize scaler
        self.scaler = self.get_scaler(experiment.get('normalization', 'None'))

        # Build layers
        self.layers = nn.ModuleList()
        self.build_layers(experiment.get('layers', []))

        # Define loss function
        self.loss_fn = self.get_loss_fn(self.loss_fn_name)

        # Initialize optimizer as None; to be set during training
        self.optimizer = None

        # Perform validation and adjust layers accordingly
        self.validate_configuration()

    def build_layers(self, layer_configs):
        MAX_CHANNELS = 1024
        previous_out_channels = None  # To track the last layer's out_channels

        for idx, layer in enumerate(layer_configs):
            layer_type_raw = layer.get('layer_type', '').strip()
            layer_type = sanitize_string(layer_type_raw).title()
            activation_fn_raw = layer.get('activation_fn', 'None')
            activation_fn_sanitized = sanitize_string(activation_fn_raw)
            activation_fn = activation_fn_sanitized.title()
            dropout_rate = layer.get('dropout_rate', None)
            layer_fields = layer.get('layer_fields', {})
            input_spec = layer.get('input')
            output_spec = layer.get('output')
            weight_init_raw = layer.get('weight_initiations', None)
            weight_init = sanitize_string(weight_init_raw) if weight_init_raw else 'none'

            if layer_type == 'Input':
                # Initialize current_shape based on input
                input_parsed = self.parse_spec(input_spec)
                if isinstance(input_parsed, tuple):
                    self.current_shape = input_parsed  # (H, W, C)
                else:
                    raise ValueError(f"Invalid input specification: {input_spec}")
                previous_out_channels = self.current_shape[2]  # Set from input channels
                continue  # Skip adding Input layer to nn.ModuleList

            elif layer_type == 'Cnn':
                if not isinstance(self.current_shape, tuple):
                    raise ValueError(
                        f"CNN layer cannot be added before an Input layer or after a Dense layer. Current shape: {self.current_shape}"
                    )

                # If 'in_channels' is specified in JSON, overwrite it with previous_out_channels
                # This ensures consistency
                # Remove 'in_channels' from layer_fields if present
                json_in_channels = layer_fields.get('in_channels')
                if 'in_channels' in layer_fields and json_in_channels != previous_out_channels:
                    print(
                        f"Layer {idx}: Overwriting 'in_channels' with {previous_out_channels} from {json_in_channels} to maintain consistency.")
                    del layer_fields['in_channels']

                in_channels = previous_out_channels
                out_channels = layer_fields.get('out_channels')

                if out_channels is None:
                    raise ValueError(
                        f"Missing 'out_channels' in layer_fields for CNN layer at index {idx}."
                    )

                # Cap the out_channels to prevent memory issues
                if out_channels > MAX_CHANNELS:
                    print(
                        f"Layer {idx}: 'out_channels' ({out_channels}) exceeds the maximum limit ({MAX_CHANNELS}). Adjusting to {MAX_CHANNELS}."
                    )
                    out_channels = MAX_CHANNELS
                    layer_fields['out_channels'] = out_channels

                kernel_size = layer_fields.get('kernel_size', 3)
                stride = layer_fields.get('stride', 1)
                padding = layer_fields.get('padding', 0)

                conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
                self.layers.append(conv)

                # Apply weight initialization if specified
                self.apply_weight_initialization(conv, weight_init, activation_fn)

                # Update current_shape
                self.current_shape = self.compute_conv_output_shape(
                    self.current_shape, kernel_size, stride, padding
                )

                previous_out_channels = out_channels  # Update tracker

            elif layer_type == 'Pooling':
                if not isinstance(self.current_shape, tuple):
                    raise ValueError(
                        f"Pooling layer cannot be added before an Input layer or after a Dense layer. Current shape: {self.current_shape}"
                    )

                pool_type_raw = layer_fields.get('pool_type', 'max')
                # Sanitize pool_type by stripping any surrounding quotes and converting to lowercase
                pool_type = sanitize_string(pool_type_raw)

                pool_size = layer_fields.get('pool_size', 2)
                stride = layer_fields.get('stride', pool_size)  # Default stride equals pool_size

                if pool_type == 'max':
                    pool = nn.MaxPool2d(kernel_size=pool_size, stride=stride)
                elif pool_type in ['average', 'avg']:
                    pool = nn.AvgPool2d(kernel_size=pool_size, stride=stride)
                else:
                    raise ValueError(f"Unsupported pool type: '{pool_type_raw}' in Pooling layer at index {idx}.")

                self.layers.append(pool)

                # Update current_shape
                self.current_shape = self.compute_pool_output_shape(
                    self.current_shape, pool_size, stride
                )

            elif layer_type == 'Batchnorm':
                if not isinstance(self.current_shape, tuple):
                    raise ValueError(
                        f"BatchNorm layer cannot be added before an Input layer or after a Dense layer. Current shape: {self.current_shape}"
                    )

                num_features = layer_fields.get('num_features', None)
                if num_features is None:
                    num_features = previous_out_channels  # Use last Conv layer's out_channels

                if isinstance(self.current_shape, tuple):
                    # Assuming [H, W, C], use BatchNorm2d
                    batch_norm = nn.BatchNorm2d(num_features)
                else:
                    # Assuming [batch_size, features], use BatchNorm1d
                    batch_norm = nn.BatchNorm1d(num_features)

                self.layers.append(batch_norm)

            elif layer_type == 'Dropout':
                rate = layer_fields.get('rate', 0.5)
                dropout = nn.Dropout(p=rate)
                self.layers.append(dropout)

            elif layer_type == 'Dense':
                # Before adding Dense layers, ensure that current_shape is flattened
                if isinstance(self.current_shape, tuple):
                    # Insert Flatten layer
                    self.layers.append(nn.Flatten())
                    input_dim = self.current_shape[0] * self.current_shape[1] * self.current_shape[2]
                    self.current_shape = input_dim  # Now it's an integer
                elif isinstance(self.current_shape, int):
                    input_dim = self.current_shape
                else:
                    raise ValueError(f"Invalid current_shape: {self.current_shape} before Dense layer.")

                units = layer_fields.get('units', None)
                if units is None:
                    raise ValueError(f"Missing 'units' in layer_fields for Dense layer at index {idx}.")

                dense = nn.Linear(input_dim, units)
                self.layers.append(dense)

                # Apply weight initialization if specified
                self.apply_weight_initialization(dense, weight_init, activation_fn)

                # Update current_shape
                self.current_shape = units

                previous_out_channels = units  # Update tracker

            elif layer_type == 'Output':
                # Handle Output layer based on loss function
                output_units_raw = layer_fields.get('output_shape', self.num_classes)
                output_units = layer_fields.get('output_shape', self.num_classes)

                if isinstance(output_units, str):
                    output_units = self.parse_spec(output_units)
                elif isinstance(output_units, (int, float)):
                    output_units = int(output_units)
                else:
                    raise ValueError(f"Invalid 'output_shape' format: {output_units}")

                # Adjust output_units based on loss function if necessary
                if self.loss_fn_name == 'Mean Squared Error':
                    # For MSE, ensure output units match num_classes (for classification)
                    output_units = self.num_classes

                # If current_shape is still a tuple, flatten it
                if isinstance(self.current_shape, tuple):
                    self.layers.append(nn.Flatten())
                    input_dim = self.current_shape[0] * self.current_shape[1] * self.current_shape[2]
                    self.current_shape = input_dim
                elif isinstance(self.current_shape, int):
                    input_dim = self.current_shape
                else:
                    raise ValueError(f"Invalid current_shape: {self.current_shape} before Output layer.")

                linear = nn.Linear(input_dim, output_units)
                self.layers.append(linear)

                # Apply weight initialization if specified
                self.apply_weight_initialization(linear, weight_init, activation_fn)

                self.current_shape = output_units

                # Adjust activation based on loss function
                if self.loss_fn_name == 'Cross Entropy Loss':
                    # For Cross Entropy, typically no activation (LogSoftmax is included in loss)
                    adjusted_activation_fn = 'None'
                elif self.loss_fn_name == 'Mean Squared Error':
                    # For MSE, you might want no activation or sigmoid depending on task
                    adjusted_activation_fn = activation_fn  # Keep as is
                elif self.loss_fn_name == 'Binary Cross Entropy':
                    # For BCE, apply Sigmoid activation
                    adjusted_activation_fn = 'Sigmoid'
                else:
                    # For other loss functions, keep the specified activation
                    adjusted_activation_fn = activation_fn

                # Add activation based on adjusted_activation_fn
                if adjusted_activation_fn != 'None':
                    activation = self.get_activation_fn(adjusted_activation_fn)
                    if activation:
                        self.layers.append(activation)

            else:
                raise ValueError(f"Unsupported layer type: {layer_type_raw} at index {idx}.")

            # Add activation function if specified and not 'None', excluding Output layer
            if layer_type not in ['Output', 'Pooling', 'Batchnorm']:
                if activation_fn != 'None':
                    activation = self.get_activation_fn(activation_fn)
                    if activation:
                        self.layers.append(activation)

            # Add dropout if specified
            if dropout_rate:
                if layer_type == 'Output':
                    print("Canceled dropout in output layer")
                else:
                    dropout = nn.Dropout(p=dropout_rate)
                    self.layers.append(dropout)

    def apply_weight_initialization(self, layer, init_method, activation_fn):
        """
            Applies the specified weight initialization to the given layer.

            Args:
                layer (nn.Module): The layer to initialize.
                init_method (str): The initialization method (e.g., 'He Initialization', 'Xavier Initialization').
                activation_fn (str): The activation function following the layer.
        """
        if init_method == 'none':
            # No initialization
            return

        # Normalize the init_method string by removing spaces and converting to lowercase
        init_method_normalized = init_method.replace(' ', '').lower()

        # Define a mapping from normalized init methods to actual PyTorch initialization functions
        init_mappings = {
            'heinitialization': 'he_normal',
            'heinit': 'he_normal',
            'xavierinitialization': 'xavier_uniform',
            'xavierinit': 'xavier_uniform',
            'normalinitialization': 'normal',
            'uniforminitialization': 'uniform',
        }

        selected_init = init_mappings.get(init_method_normalized, None)

        if selected_init is None:
            print(f"Warning: Unknown initialization method '{init_method}'. Using default initialization.")
            return  # Use default initialization

        try:
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                if selected_init == 'he_normal':
                    # Determine nonlinearity based on activation function
                    if activation_fn.lower() == 'leakyrelu':
                        nonlinearity = 'leaky_relu'
                    else:
                        nonlinearity = 'relu'
                    nn.init.kaiming_normal_(layer.weight, nonlinearity=nonlinearity)
                elif selected_init == 'xavier_uniform':
                    nn.init.xavier_uniform_(layer.weight)
                elif selected_init == 'normal':
                    nn.init.normal_(layer.weight, mean=0.0, std=0.02)
                elif selected_init == 'uniform':
                    nn.init.uniform_(layer.weight, a=-0.05, b=0.05)

                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        except Exception as e:
            print(f"Error initializing layer {layer}: {e}")

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
        """
            Retrieves the corresponding PyTorch activation function based on the provided name.
            Handles case insensitivity and different formatting styles.

            Args:
                name (str): The name of the activation function.

            Returns:
                nn.Module or None: The PyTorch activation function module or None if 'None' is specified.

            Raises:
                ValueError: If the activation function name is unsupported.
        """
        if not isinstance(name, str):
            raise TypeError(f"Activation function name must be a string, got {type(name)}.")

        # Normalize the activation function name: remove spaces and underscores, convert to lowercase
        name_normalized = name.replace(' ', '').replace('_', '').lower()

        # Define activation functions with normalized keys
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'softmax': nn.Softmax(dim=1),
            'leakyrelu': nn.LeakyReLU(negative_slope=0.01),
            # Add more activations as needed
        }

        activation = activations.get(name_normalized, None)
        if activation is None and name_normalized != 'none':
            raise ValueError(
                f"Unsupported activation function: '{name}'. Supported functions: {list(activations.keys()) + ['None']}")
        return activation

    def get_loss_fn(self, name):
        """
            Retrieves the corresponding PyTorch loss function based on the provided name.

            Args:
                name (str): The name of the loss function.

            Returns:
                nn.Module: The PyTorch loss function module.

            Raises:
                ValueError: If the loss function name is unsupported.
        """
        # Updated to match the loss_fn names in your JSON
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
        """
            Retrieves the corresponding PyTorch optimizer based on the configuration.

            Args:
                parameters (iterable): The parameters to optimize.

            Returns:
                torch.optim.Optimizer: The PyTorch optimizer.

            Raises:
                ValueError: If the optimizer name is unsupported.
        """
        optimizers = {
            'Adam': optim.Adam,
            'Sgd': optim.SGD,
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
        elif self.optimization == 'Sgd':
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
        """
            Retrieves the corresponding scaler based on the provided name.

            Args:
                name (str): The name of the scaler.

            Returns:
                scaler or None: The scaler object or None if 'None' is specified.

            Raises:
                ValueError: If the scaler name is unsupported.
        """
        scalers = {
            'standardscaler': StandardScaler(),
            'minmaxscaler': MinMaxScaler(),
            'normalizer': Normalizer(),
            'none': None
        }
        name_normalized = name.replace(' ', '').replace('_', '').lower()
        scaler = scalers.get(name_normalized, None)
        if scaler is None and name_normalized != 'none':
            raise ValueError(
                f"Unsupported normalization: '{name}'. Supported scalers: {list(scalers.keys()) + ['None']}")
        return scaler

    def forward(self, x):
        """
            Defines the forward pass of the model.

            Args:
                x (torch.Tensor): The input tensor.

            Returns:
                torch.Tensor: The output tensor.
        """
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

            Args:
                x (numpy.ndarray): The input data.

            Returns:
                torch.Tensor: The preprocessed tensor.
        """
        if self.scaler:
            # Assuming x is a NumPy array; convert to tensor after scaling
            x = self.scaler.transform(x)
            x = torch.tensor(x, dtype=torch.float32)
        else:
            x = torch.tensor(x, dtype=torch.float32)
        return x

    def validate_configuration(self):
        """
            Validates the model configuration to ensure compatibility between
            the loss function, activation functions, and output layer.
            Automatically adjusts activation functions to match loss function expectations.
        """
        # Mapping of loss functions to expected output activation and label format
        loss_fn_config = {
            'Cross Entropy Loss': {
                'activation': None,  # No activation; CrossEntropyLoss applies LogSoftmax
                'label_format': 'class_indices'  # Labels should be class indices (LongTensor)
            },
            'Binary Cross Entropy': {
                'activation': 'Sigmoid',  # Sigmoid activation for BCE
                'label_format': 'binary'  # Labels should be binary (FloatTensor)
            },
            'Mean Squared Error': {
                'activation': None,  # Typically no activation; depends on task
                'label_format': 'regression'  # Labels should match output shape
            },
            # Add more configurations as needed
        }

        current_loss_fn = self.loss_fn_name
        if current_loss_fn not in loss_fn_config:
            raise ValueError(f"Unsupported loss function for validation: {current_loss_fn}")

        expected_activation = loss_fn_config[current_loss_fn]['activation']
        label_format = loss_fn_config[current_loss_fn]['label_format']

        # Determine the last layer's activation function
        last_layer = self.layers[-1]
        if isinstance(last_layer, nn.Linear):
            last_activation = None
        else:
            # Iterate backwards to find the last activation function
            last_activation = None
            for layer in reversed(self.layers):
                if isinstance(layer, (nn.ReLU, nn.Tanh, nn.Sigmoid, nn.Softmax, nn.LeakyReLU)):
                    last_activation = layer.__class__.__name__
                    break

        # Specific handling based on loss function
        if current_loss_fn == 'Cross Entropy Loss':
            # Automatically remove any activation function after Output layer
            if last_activation is not None:
                # Find the index of the last activation layer
                for i in reversed(range(len(self.layers))):
                    if isinstance(self.layers[i], (nn.ReLU, nn.Tanh, nn.Sigmoid, nn.Softmax, nn.LeakyReLU)):
                        # Remove the activation layer
                        print(f"Removing incompatible activation function '{self.layers[i].__class__.__name__}' from Output layer.")
                        del self.layers[i]
                        break
            # Ensure no activation function is present after the Output layer
            # No need to raise an error; adjustments have been made

        elif current_loss_fn == 'Mean Squared Error':
            # Ensure the final layer outputs 'num_classes' units
            if not isinstance(last_layer, nn.Linear) or last_layer.out_features != self.num_classes:
                # Remove the incorrect final layer if present
                if not isinstance(last_layer, nn.Linear):
                    print(f"Removing non-linear Output layer '{last_layer}' for MSE Loss.")
                    self.layers.pop(-1)
                elif last_layer.out_features != self.num_classes:
                    print(f"Adjusting Output layer from {last_layer.out_features} to {self.num_classes} units for MSE Loss.")
                    self.layers.pop(-1)

                # Determine input size for the new Linear layer
                if isinstance(self.current_shape, tuple):
                    input_dim = self.current_shape[0] * self.current_shape[1] * self.current_shape[2]
                else:
                    input_dim = self.current_shape

                # Add a new Linear layer with 'num_classes' outputs
                linear = nn.Linear(input_dim, self.num_classes)
                self.layers.append(linear)
                self.current_shape = self.num_classes
                print(f"Adjusted the final Linear layer to have {self.num_classes} output units for MSE Loss.")

            # Optionally, check for activation function (typically none for MSE)
            if expected_activation:
                if last_activation != expected_activation:
                    activation = self.get_activation_fn(expected_activation)
                    if activation:
                        self.layers.append(activation)
                        print(f"Added activation function '{expected_activation}' to the model.")

        elif current_loss_fn == 'Binary Cross Entropy':
            # Ensure the final activation function is Sigmoid
            if last_activation != 'Sigmoid':
                # Check if an activation function exists after Output layer
                for i in reversed(range(len(self.layers))):
                    if isinstance(self.layers[i], nn.Linear):
                        # Insert Sigmoid activation after the Output layer
                        print(f"Adding 'Sigmoid' activation after Output layer for BCE Loss.")
                        self.layers.insert(i + 1, nn.Sigmoid())
                        break
        else:
            # Handle other loss functions as needed
            if expected_activation:
                if last_activation != expected_activation:
                    # Find the last activation layer and replace it
                    for i in reversed(range(len(self.layers))):
                        if isinstance(self.layers[i], (nn.ReLU, nn.Tanh, nn.Sigmoid, nn.Softmax, nn.LeakyReLU)):
                            print(f"Replacing activation function '{self.layers[i].__class__.__name__}' with '{expected_activation}'.")
                            self.layers[i] = self.get_activation_fn(expected_activation)
                            break
            else:
                if last_activation is not None and last_activation != 'Softmax':
                    # Remove the activation function if it's not expected
                    for i in reversed(range(len(self.layers))):
                        if isinstance(self.layers[i], (nn.ReLU, nn.Tanh, nn.Sigmoid, nn.Softmax, nn.LeakyReLU)):
                            print(f"Removing incompatible activation function '{self.layers[i].__class__.__name__}' as it's not expected by the loss function.")
                            del self.layers[i]
                            break

        # Validate output layer
        output_layer = self.layers[-1]
        if isinstance(output_layer, nn.Linear):
            output_units = output_layer.out_features
            if label_format == 'class_indices' and output_units < 2:
                raise ValueError(
                    f"Loss Function '{current_loss_fn}' expects at least 2 output units for multi-class classification."
                )
            # Additional checks can be added based on label_format
        else:
            raise ValueError("The last layer must be a Linear (Output) layer.")

        # Informative message upon successful validation
        print("Model configuration validated successfully.")

def get_DynamicModel(json_input):
    ret_model = DynamicModel(json_input)
    ret_model.initialize_optimizer()

    return ret_model
