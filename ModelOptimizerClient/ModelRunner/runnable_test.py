import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
from datetime import datetime
from ModelRunner.DynamicModel import get_DynamicModel

torch.manual_seed(0) # to make sure all tests are more or else the same...


def train_model(test_config, train_dataset, test_dataset):
    """
    Trains the DynamicModel based on the provided test configuration and datasets.

    Parameters:
    - test_config (dict): Dictionary containing the entire test configuration, including 'exp_id' and 'test_id'.
    - train_dataset (torch.utils.data.Dataset): Training dataset.
    - test_dataset (torch.utils.data.Dataset): Testing dataset.

    Returns:
    - result_json (str): JSON string containing training and testing statistics or error information.
    """
    try:
        # Initialize the model
        model = get_DynamicModel(test_config)
    except Exception as e:
        # Handle errors during model initialization
        result = {
            "status": "Failed",
            "test_id": test_config.get('test_id', None),
            "exp_id": test_config.get('exp_id', None),
            "error_message": f"Model Initialization Error: {str(e)}",
            "execution_timestamp": datetime.now().isoformat()
        }
        return json.dumps(result, indent=4)

    # Determine the device to use (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'

    model.to(device)

    # Extract training parameters
    experiment = test_config.get('experiment_data', {})
    batch_size = experiment.get('batch_size', 32)
    epochs = experiment.get('epochs', 10)
    loss_fn_name = experiment.get('loss_fn', 'Cross Entropy Loss')

    # Define DataLoaders for training and testing
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define the loss function and optimizer
    criterion = model.loss_fn  # Already defined in DynamicModel
    optimizer = model.optimizer  # Already initialized in DynamicModel

    # Lists to store training statistics
    epoch_losses = []
    epoch_accuracies = []
    last_trained_accuracy = 0
    # Start tracking training time
    start_time = time.time()

    # Training loop
    try:
        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            correct_predictions = 0
            total_samples = 0

            for batch_idx, (inputs, labels) in enumerate(train_loader):
                # Move inputs and labels to the device
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs)

                # Compute loss
                loss = criterion(outputs, labels)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                # Accumulate loss
                running_loss += loss.item() * inputs.size(0)

                # Compute accuracy based on loss function
                if loss_fn_name == 'Cross Entropy Loss':
                    _, predicted = torch.max(outputs.data, 1)
                    correct_predictions += (predicted == labels).sum().item()
                elif loss_fn_name == 'Binary Cross Entropy':
                    # Assuming binary classification with outputs as logits
                    predicted = (torch.sigmoid(outputs) > 0.5).float()
                    correct_predictions += (predicted == labels).sum().item()
                else:
                    # Handle other loss functions or set accuracy to None
                    pass

                total_samples += labels.size(0)

            # Calculate average loss and accuracy for the epoch
            epoch_loss = running_loss / total_samples
            if loss_fn_name in ['Cross Entropy Loss', 'Binary Cross Entropy']:
                epoch_accuracy = correct_predictions / total_samples
                epoch_accuracies.append(round(epoch_accuracy, 5))  # Rounded for JSON compatibility
                print(f"Epoch {epoch}/{epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_accuracy:.4f}")
            else:
                epoch_accuracy = None  # Not applicable
                print(f"Epoch {epoch}/{epochs} - Loss: {epoch_loss:.4f}")

            last_trained_accuracy = epoch_accuracy
            epoch_losses.append(epoch_loss)

    except Exception as e:
        # Handle errors during training
        end_time = time.time()
        training_time_seconds = end_time - start_time
        training_date = datetime.now().isoformat()

        if epoch_losses:
            # Partial completion
            result = {
                "status": "Partial",
                "test_id": test_config.get('test_id', None),
                "exp_id": test_config.get('exp_id', None),
                "device_name": device_name,
                "error_message": f"Training Error after {epoch} epochs: {str(e)}",
                "train_stats": {
                    "epoch_losses": [round(loss, 5) for loss in epoch_losses],
                    "epoch_accuracies": epoch_accuracies,
                    "final_loss": round(epoch_losses[-1], 5),
                    "final_accuracy": epoch_accuracies[-1] if epoch_accuracy is not None else None,
                    "epochs_trained": epoch,
                    "training_time_seconds": round(training_time_seconds, 5),
                    "training_date": training_date
                },
                "test_stats": None,
                "model_architecture": str(model)
            }
        else:
            # No completion
            result = {
                "status": "Failed",
                "test_id": test_config.get('test_id', None),
                "exp_id": test_config.get('exp_id', None),
                "device_name": device_name,
                "error_message": f"Training Error: {str(e)}",
                "execution_timestamp": datetime.now().isoformat()
            }
        return json.dumps(result, indent=4)

    # Record training time
    end_time = time.time()
    training_time_seconds = end_time - start_time

    # Get the current date and time in ISO format
    training_date = datetime.now().isoformat()

    # Evaluation on the test dataset
    model.eval()
    test_loss = 0.0
    correct_test_predictions = 0
    total_test_samples = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            # Move inputs and labels to the device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)

            # Predictions based on loss function
            if loss_fn_name == 'Cross Entropy Loss':
                _, predicted = torch.max(outputs.data, 1)
                correct_test_predictions += (predicted == labels).sum().item()
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            elif loss_fn_name == 'Binary Cross Entropy':
                # Assuming binary classification with outputs as logits
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                correct_test_predictions += (predicted == labels).sum().item()
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            else:
                # Handle other loss functions or set predictions to None
                pass

            total_test_samples += labels.size(0)

    # Calculate average loss and accuracy for the test dataset
    final_loss = test_loss / total_test_samples
    if loss_fn_name in ['Cross Entropy Loss', 'Binary Cross Entropy']:
        final_accuracy = correct_test_predictions / total_test_samples
    else:
        final_accuracy = None  # Not applicable

    # Compute additional test statistics
    if len(all_predictions) > 0 and len(all_labels) > 0:
        all_predictions_tensor = torch.tensor(all_predictions, dtype=torch.float32, device=device)
        all_labels_tensor = torch.tensor(all_labels, dtype=torch.float32, device=device)

        # Mean Squared Error (MSE)
        mse = nn.MSELoss()(all_predictions_tensor, all_labels_tensor).item()

        # Variance of the dataset labels
        variance_dataset = float(all_labels_tensor.var().item())

        # Variance of the predictions
        variance_y_hat = float(all_predictions_tensor.var().item())

        # Mean Bias (mean of predictions - mean of labels)
        mean_bias = float(all_predictions_tensor.mean().item() - all_labels_tensor.mean().item())

        # Compile test statistics
        test_stats = {
            "mse": mse,
            "variance_dataset": variance_dataset,
            "variance_y_hat": variance_y_hat,
            "mean_bias": mean_bias,
            "accuracy": round(final_accuracy, 5) if final_accuracy is not None else None
        }
    else:
        # Handle cases where predictions or labels are empty
        test_stats = {
            "mse": None,
            "variance_dataset": None,
            "variance_y_hat": None,
            "mean_bias": None,
            "accuracy": round(final_accuracy, 5) if final_accuracy is not None else None
        }

    # Get the model architecture as a string
    model_architecture = str(model)

    # Compile training statistics
    train_stats = {
        "epoch_losses": [round(loss, 5) for loss in epoch_losses],
        "epoch_accuracies": epoch_accuracies,
        "final_loss": round(final_loss, 5),
        "final_accuracy": round(last_trained_accuracy, 5) if last_trained_accuracy is not None else None,
        "epochs_trained": epochs,
        "training_time_seconds": round(training_time_seconds, 5),
        "training_date": training_date
    }

    # Prepare the final JSON result
    result = {
        "status": "Success",
        "test_id": test_config.get('test_id', None),
        "exp_id": test_config.get('exp_id', None),
        "device_name": device_name,
        "train_stats": train_stats,
        "test_stats": test_stats,
        "model_architecture": model_architecture
    }

    # Convert the result dictionary to a JSON string
    result_json = json.dumps(result, indent=4)

    return result_json
