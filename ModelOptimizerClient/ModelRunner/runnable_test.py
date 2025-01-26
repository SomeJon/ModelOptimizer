import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from datetime import datetime

from ModelRunner.DynamicModel import get_DynamicModel


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

                # Compute accuracy
                _, predicted = torch.max(outputs.data, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)

            # Calculate average loss and accuracy for the epoch
            epoch_loss = running_loss / total_samples
            epoch_accuracy = correct_predictions / total_samples

            epoch_losses.append(epoch_loss)
            epoch_accuracies.append(round(epoch_accuracy, 5))  # Rounded for JSON compatibility

            print(f"Epoch {epoch}/{epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_accuracy:.4f}")

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
                    "final_accuracy": epoch_accuracies[-1],
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

            # Predictions
            _, predicted = torch.max(outputs.data, 1)
            correct_test_predictions += (predicted == labels).sum().item()
            total_test_samples += labels.size(0)

            # Collect all predictions and labels for additional metrics
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate average loss and accuracy for the test dataset
    final_loss = test_loss / total_test_samples
    final_accuracy = correct_test_predictions / total_test_samples

    # Compute additional test statistics
    # Mean Squared Error (MSE)
    mse = nn.MSELoss()(torch.tensor(all_predictions), torch.tensor(all_labels)).item()

    # Variance of the dataset labels
    variance_dataset = float(torch.tensor(all_labels).var().item())

    # Variance of the predictions
    variance_y_hat = float(torch.tensor(all_predictions).var().item())

    # Mean Bias (mean of predictions - mean of labels)
    mean_bias = float(torch.tensor(all_predictions).float().mean().item() - torch.tensor(all_labels).float().mean().item())

    # Accuracy (already computed as final_accuracy)

    # Compile test statistics
    test_stats = {
        "mse": mse,
        "variance_dataset": variance_dataset,
        "variance_y_hat": variance_y_hat,
        "mean_bias": mean_bias,
        "accuracy": round(final_accuracy, 5)
    }

    # Get the model architecture as a string
    model_architecture = str(model)

    # Compile training statistics
    train_stats = {
        "epoch_losses": [round(loss, 5) for loss in epoch_losses],
        "epoch_accuracies": epoch_accuracies,
        "final_loss": round(final_loss, 5),
        "final_accuracy": round(final_accuracy, 5),
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
