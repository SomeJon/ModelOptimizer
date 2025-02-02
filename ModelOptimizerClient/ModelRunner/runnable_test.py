import json
import torch
import time
from datetime import datetime
from ModelRunner.DynamicModel import get_DynamicModel
from ModelRunner.dataset_utils import get_data_loader
from ModelRunner.get_cifar_dataset import get_cifar10_datasets

torch.manual_seed(0) # to make sure all tests are more or less the same...


def train_model(test_config, train_dataset, valid_dataset, test_dataset):
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
    min_delta = experiment.get('min_delta', None)
    patience = experiment.get('patience', None)

    train_loader = get_data_loader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        use_cuda=torch.cuda.is_available()
    )
    test_loader = get_data_loader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        use_cuda=torch.cuda.is_available()
    )
    valid_loader = get_data_loader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        use_cuda=torch.cuda.is_available()
    )

    early_stopping = EarlyStopping(patience=patience, delta=min_delta)

    # Lists to store training statistics
    epoch_losses_train = []
    epoch_accuracies_train = []
    epoch_losses_validation = []
    epoch_accuracies_validation = []
    last_trained_accuracy = 0
    # Start tracking training time
    start_time = time.time()
    epoch_trained = 0
    # Training loop
    try:
        for epoch in range(1, epochs + 1):
            model.train()
            epoch_accuracy, train_epoch_loss = run_epoch(device, train_loader, loss_fn_name, model, epoch, epochs, 'Train', True)
            if epoch_accuracy is not None:
                epoch_accuracies_train.append(round(epoch_accuracy, 5))
            epoch_losses_train.append(train_epoch_loss)
            last_trained_accuracy = epoch_accuracy
            epoch_trained += 1
            model.eval()
            epoch_accuracy, valid_epoch_loss = run_epoch(device, valid_loader, loss_fn_name, model, epoch, epochs, 'Validation', False)
            if epoch_accuracy is not None:
                epoch_accuracies_validation.append(round(epoch_accuracy, 5))
            epoch_losses_validation.append(valid_epoch_loss)

            if early_stopping(valid_epoch_loss):  # Check early stopping
                print("Early stopping triggered!")
                break

    except Exception as e:
        # Handle errors during training
        end_time = time.time()
        training_time_seconds = end_time - start_time
        training_date = datetime.now().isoformat()

        if epoch_losses_train:
            # Partial completion
            result = {
                "status": "Partial",
                "test_id": test_config.get('test_id', None),
                "exp_id": test_config.get('exp_id', None),
                "device_name": device_name,
                "error_message": f"Training Error after {epoch} epochs: {str(e)}",
                "train_stats": {
                    "epoch_losses_train": [round(loss, 5) for loss in epoch_losses_train],
                    "epoch_accuracies_train": epoch_accuracies_train,
                    "epoch_losses_validation": epoch_losses_validation,
                    "epoch_accuracies_validation": epoch_accuracies_validation,
                    "final_loss": round(epoch_losses_train[-1], 5),
                    "final_accuracy": last_trained_accuracy if last_trained_accuracy is not None else None,
                    "epochs_trained": epoch_trained,
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
    epoch_accuracy, epoch_loss = run_epoch(device, test_loader, loss_fn_name, model, 1, 1, 'Test', True)

    test_stats = {
        "test_loss": epoch_loss,
        "test_accuracy": round(epoch_accuracy, 5) if epoch_accuracy is not None else None
    }

    # Get the model architecture as a string
    model_architecture = str(model)

    # Compile training statistics
    train_stats = {
        "epoch_losses_train": [round(loss, 5) for loss in epoch_losses_train],
        "epoch_accuracies_train": epoch_accuracies_train,
        "epoch_losses_validation": epoch_losses_validation,
        "epoch_accuracies_validation": epoch_accuracies_validation,
        "final_loss": round(epoch_losses_train[-1], 5),
        "final_accuracy": last_trained_accuracy if last_trained_accuracy is not None else None,
        "epochs_trained": epoch_trained,
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


def get_labels(labels, loss_fn_name, device, num_classes=10):
    """
    Processes labels based on the loss function.

    Parameters:
    - labels (Tensor): Original labels.
    - loss_fn_name (str): Name of the loss function.
    - device (torch.device): Device to move labels to.
    - num_classes (int): Number of classes for one-hot encoding.

    Returns:
    - Tensor: Processed labels.
    """
    if loss_fn_name == 'Mean Squared Error':
        # Ensure labels are of type Long for scatter_
        labels = labels.to(device).long().unsqueeze(1)  # Shape: [batch_size, 1]
        # Initialize one-hot encoded tensor
        one_hot = torch.zeros(labels.size(0), num_classes).to(device)
        # Scatter 1s at the appropriate indices
        one_hot.scatter_(1, labels, 1)
        # Convert to float for MSELoss
        labels = one_hot.float()
    elif loss_fn_name == 'Cross Entropy Loss':
        labels = labels.to(device).long()
    elif loss_fn_name == 'Binary Cross Entropy':
        labels = labels.to(device).float()
    else:
        labels = labels.to(device).float()
    return labels


def run_epoch(device, data_loader, loss_fn_name, model, epoch, epochs, print_label='Train', prints=False, num_classes=10):
    """
    Runs one epoch of training or validation.

    Parameters:
    - device (torch.device): Device to run the computations on.
    - data_loader (DataLoader): DataLoader for the dataset.
    - loss_fn_name (str): Name of the loss function.
    - model (nn.Module): The model to train/evaluate.
    - epoch (int): Current epoch number.
    - epochs (int): Total number of epochs.
    - num_classes (int): Number of classes for one-hot encoding (default: 10).

    Returns:
    - tuple: (epoch_accuracy, epoch_loss)
    """
    criterion = model.loss_fn  # Already defined in DynamicModel
    optimizer = model.optimizer  # Already initialized in DynamicModel

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.set_grad_enabled(model.training):
        for batch_idx, (inputs, labels) in enumerate(data_loader):
            # Move inputs and labels to the device
            inputs = inputs.to(device)
            labels = get_labels(labels, loss_fn_name, device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # For MSELoss, apply Softmax to outputs if using one-hot labels
            if loss_fn_name == 'Mean Squared Error':
                outputs = torch.softmax(outputs, dim=1)

            # Compute loss
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            if model.training:
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
            elif loss_fn_name == 'Mean Squared Error':
                # Assuming one-hot labels
                _, predicted = torch.max(outputs.data, 1)
                _, actual = torch.max(labels.data, 1)
                correct_predictions += (predicted == actual).sum().item()
            else:
                pass

            total_samples += labels.size(0)

    # Calculate average loss and accuracy for the epoch
    epoch_loss = running_loss / total_samples
    if loss_fn_name in ['Cross Entropy Loss', 'Binary Cross Entropy', 'Mean Squared Error']:
        epoch_accuracy = correct_predictions / total_samples
        if prints:
            print(f"-{print_label}- Epoch {epoch}/{epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_accuracy:.4f}")
    else:
        epoch_accuracy = None  # Not applicable
        if prints:
            print(f"-{print_label}- Epoch {epoch}/{epochs} - Loss: {epoch_loss:.4f}")

    return epoch_accuracy, epoch_loss


class EarlyStopping:
    def __init__(self, patience=None, delta=None):
        self.patience = patience
        self.delta = delta if delta is not None else 0  # Default delta to 0 if not specified
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        # If patience is None, always return False (no early stopping)
        if self.patience is None:
            self.early_stop = False
            return False

        # Standard early stopping logic
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

        return self.early_stop


def run_tests(selected_tests, completed_tests):
    for idx, test in enumerate(selected_tests, 1):
        print(f"\nExecuting Test {idx}/{len(selected_tests)}:")
        exp_id = test.get('exp_id', 'Unknown')
        test_id = test.get('test_id', 'Unknown')
        print(f"Exp ID: {exp_id}, Test ID: {test_id}")

        # Extract necessary information from the test configuration
        experiment_data = test.get('experiment_data', {})
        if not experiment_data:
            print(f"Error: Test {exp_id}-{test_id} is missing 'experiment_data'. Moving to results as Failed.")
            # Create a failed result entry
            result = {
                "status": "Failed",
                "test_id": test.get('test_id', None),
                "exp_id": test.get('exp_id', None),
                "error_message": "'experiment_data' field is missing in the JSON configuration.",
                "execution_timestamp": datetime.now().isoformat()
            }
            completed_tests.append(result)
            continue

        # Ensure experiment_data is a dictionary
        if not isinstance(experiment_data, dict):
            print(f"Error: Test {exp_id}-{test_id} has invalid 'experiment_data' format. Moving to results as Failed.")
            # Create a failed result entry
            result = {
                "status": "Failed",
                "test_id": test.get('test_id', None),
                "exp_id": test.get('exp_id', None),
                "error_message": "'experiment_data' is not a valid dictionary.",
                "execution_timestamp": datetime.now().isoformat()
            }
            completed_tests.append(result)
            continue

        # Pass the entire test dictionary to train_model
        try:
            train_dataset, valid_dataset, test_dataset = get_cifar10_datasets(experiment_data.get('normalization', None))

            result_json = train_model(test, train_dataset, valid_dataset, test_dataset)
            result = json.loads(result_json)

            # Check the status and handle accordingly
            status = result.get('status', 'Failed')
            if status == 'Success':
                print(f"Test {exp_id}-{test_id} completed successfully.")
            elif status == 'Partial':
                print(f"Test {exp_id}-{test_id} completed partially.")
            elif status == 'Failed':
                print(f"Test {exp_id}-{test_id} failed: {result.get('error_message', 'Unknown error.')}")
            else:
                print(f"Test {exp_id}-{test_id} returned an unknown status: {status}")

            # Append the result to completed_tests regardless of status
            completed_tests.append(result)

        except json.JSONDecodeError:
            print(f"Error: Received invalid JSON response for Test {exp_id}-{test_id}. Moving to results as Failed.")
            # Create a failed result entry
            result = {
                "status": "Failed",
                "test_id": test.get('test_id', None),
                "exp_id": test.get('exp_id', None),
                "error_message": "Invalid JSON response from train_model.",
                "execution_timestamp": datetime.now().isoformat()
            }
            completed_tests.append(result)
        except Exception as e:
            print(f"Error executing Test {exp_id}-{test_id}: {e}. Moving to results as Failed.")
            # Create a failed result entry
            result = {
                "status": "Failed",
                "test_id": test.get('test_id', None),
                "exp_id": test.get('exp_id', None),
                "error_message": f"Unexpected error: {str(e)}",
                "execution_timestamp": datetime.now().isoformat()
            }
            completed_tests.append(result)

    # Determine which tests were successfully run (Success or Partial)
    successfully_run_tests = [test for test in selected_tests if any(
        (result.get('test_id') == test.get('test_id') and result.get('exp_id') == test.get('exp_id'))
        for result in completed_tests
    )]

    return completed_tests, successfully_run_tests
