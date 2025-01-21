from dto.LayerDto import LayerDto
from utils import DB
import pymysql


class ModelDto:
    def __init__(self, model_id, database_id, loss_fn, optimization, normalization, batch_size, weight_decay,
                 learning_rate, layers):
        self.model_id = model_id
        self.database_id = database_id
        self.loss_fn = loss_fn
        self.optimization = optimization
        self.normalization = normalization
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.layers = layers  # List of LayerDto objects

    def to_dict(self):
        return {
            "model_id": self.model_id,
            "database_id": self.database_id,
            "loss_fn": self.loss_fn,
            "optimization": self.optimization,
            "normalization": self.normalization,
            "batch_size": self.batch_size,
            "weight_decay": self.weight_decay,
            "learning_rate": self.learning_rate,
            "layers": [layer.to_dict() for layer in self.layers],
        }

    def __str__(self):
        layers_str = "\n".join([str(layer) for layer in self.layers])
        return f"Model ID: {self.model_id}, Loss: {self.loss_fn}, Layers:\n{layers_str}"


def get_model_by_id(model_id):
    """
    Fetches a model from the database by its ID and returns a ModelDto object.
    :param model_id: The ID of the model to fetch.
    :return: ModelDto object if the model exists, otherwise None.
    """
    connection = None
    try:
        # Get database connection
        connection = DB.get_connection()
        cursor = connection.cursor(pymysql.cursors.DictCursor)

        # Fetch model details
        cursor.execute("SELECT * FROM model WHERE model_id = %s", (model_id,))
        model_data = cursor.fetchone()

        if not model_data:
            return None  # Model not found

        # Fetch layer information from `model_layer`
        cursor.execute("SELECT * FROM model_layer WHERE model_id = %s ORDER BY layer_place", (model_id,))
        model_layers_data = cursor.fetchall()

        if not model_layers_data:
            return None  # No layers found for the model

        # Fetch all related layer details from `layer`
        layer_ids = [layer["layer_id"] for layer in model_layers_data]
        cursor.execute(f"SELECT * FROM layer WHERE layer_id IN ({', '.join(['%s'] * len(layer_ids))})", layer_ids)
        layers_data = cursor.fetchall()

        # Map layer details by layer_id for quick lookup
        layers_dict = {layer["layer_id"]: layer for layer in layers_data}

        # Construct LayerDto objects in the correct order
        layers = [
            LayerDto(
                layer_id=layer["layer_id"],
                layer_type=layer["layer_type"],
                activation_fn=layer["activation_fn"],
                weight_initiations=layer["weight_initiations"],
                input_size=layer["input"],
                output_size=layer["output"],
                metadata=layer.get("meta_data")
            )
            for layer in (layers_dict[model_layer["layer_id"]] for model_layer in model_layers_data)
        ]

        # Construct and return ModelDto
        return ModelDto(
            model_id=model_data["model_id"],
            name=model_data["name"],
            layers=layers,
            metadata=model_data.get("metadata")
        )

    except Exception as e:
        print(f"Error fetching model: {e}")
        raise
    finally:
        if connection:
            connection.close()