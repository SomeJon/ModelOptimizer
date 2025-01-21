class LayerDto:
    def __init__(self, layer_id, layer_type, activation_fn, weight_initiations, input_size, output_size, dropout_rate, metadata):
        self.layer_id = layer_id
        self.layer_type = layer_type
        self.activation_fn = activation_fn
        self.weight_initiations = weight_initiations
        self.input_size = input_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.metadata = metadata

    def to_dict(self):
        return {
            "layer_id": self.layer_id,
            "layer_type": self.layer_type,
            "activation_fn": self.activation_fn,
            "weight_initiations": self.weight_initiations,
            "input_size": self.input_size,
            "output_size": self.output_size,
            "dropout_rate": self.dropout_rate,
            "metadata": self.metadata,
        }

    def __str__(self):
        return f"Layer ID: {self.layer_id}, Type: {self.layer_type}, Activation: {self.activation_fn}"
