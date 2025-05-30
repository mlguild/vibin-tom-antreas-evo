# Gradient-Based Training Process

This document describes the gradient-based training pipeline, orchestrated by `run_gradient_based.py` and utilizing helper functions from `tinkering/train.py`.

## 1. Orchestration Script: `run_gradient_based.py`

This script serves as the main entry point for training the `SimpleResNet` model on the EMNIST dataset using traditional gradient-based optimization.

### 1.1. Command-Line Interface (CLI)

Uses `python-fire` to expose the `main` function as a CLI.
Key configurable parameters include:
*   `--num_total_iterations`: Total number of training steps (default: 20000).
*   `--lr`: Initial learning rate for AdamW (default: 1e-3, user modified to 1e-5).
*   `--weight_decay`: Weight decay for AdamW (default: 1e-4).
*   `--batch_size`: Batch size for DataLoaders (default: 64).
*   `--emnist_split`: EMNIST split to use (default: 'byclass').
*   `--data_dir_root`: Root directory for EMNIST data (default: './emnist_data').
*   `--eval_interval_iters`: How often to run validation (default: 500 iterations).
*   `--log_interval_iters`: How often to log training metrics (default: 100 iterations).
*   `--num_workers`: Number of workers for DataLoaders (default: 4).
*   `--save_path`: Directory to save model checkpoints (default: './saved_models').
*   `--model_name`: Filename for the saved model (default: 'resnet_emnist.pt').
*   `--final_lr_factor`: Factor by which the initial LR is multiplied at the end of training (default: 0.01, for 1/100th).

### 1.2. Core Operations in `main` function:
1.  **Setup**: 
    *   Installs `rich.traceback` for better error reporting.
    *   Sets up the device (`cuda` or `cpu`).
    *   Prepares data and model save paths.
2.  **Dataloaders**: Calls `get_emnist_dataloaders` from `tinkering.emnist_dataset`.
3.  **Model**: Initializes `ResNet4StageCustom` from `tinkering.models` and moves it to the device.
4.  **Optimizer**: Uses `torch.optim.AdamW`.
5.  **Loss Function**: Uses `torch.nn.CrossEntropyLoss`.
6.  **Learning Rate Scheduler**: Implements `torch.optim.lr_scheduler.LambdaLR` for a linear decay of the learning rate. The rate decreases from its initial value to `initial_lr * final_lr_factor` over `num_total_iterations`.
7.  **Training Loop**: 
    *   Iterates for `num_total_iterations`.
    *   Uses `rich.progress.Progress` for an overall progress bar.
    *   Calls `run_train_iterations` for segments of `eval_interval_iters`.
    *   Calls `run_validation` after each training segment.
    *   Logs training and validation metrics using `rich.console`.
    *   Saves the model checkpoint to `model_save_path` whenever a better validation accuracy is achieved.

## 2. Training and Validation Logic: `tinkering/train.py`

This module contains the core functions for performing training steps and validation.

### 2.1. `run_train_iterations(...)

```python
def run_train_iterations(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    num_train_iterations: int,
    current_step: int = 0,
    scheduler: torch.optim.lr_scheduler._LRScheduler = None,
) -> tuple:
    # ... implementation ...
```
*   Sets the model to `train()` mode.
*   Takes `num_train_iterations` as an argument.
*   Uses `itertools.cycle(dataloader)` to allow training for more iterations than batches in one epoch.
*   Iterates for the specified number of training steps:
    *   Fetches a batch of data.
    *   Moves data to the device.
    *   Performs optimizer zero_grad, forward pass, loss calculation, backward pass, and optimizer step.
    *   If a `scheduler` is provided, calls `scheduler.step()` after each optimizer step.
    *   Tracks per-batch loss and accuracy in lists.
    *   Updates a `tqdm` progress bar with current loss, accuracy, and learning rate.
*   Returns the average loss, average accuracy over the processed batches, and the updated global step count.

### 2.2. `run_validation(...)

```python
def run_validation(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    # ... implementation ...
```
*   Sets the model to `eval()` mode.
*   Iterates through the *entire* validation dataloader once.
*   Uses `torch.no_grad()` to disable gradient calculations.
*   Calculates loss and accuracy for each batch.
*   Tracks per-batch loss and accuracy in lists.
*   Updates a `tqdm` progress bar.
*   Returns the average loss and average accuracy over the entire validation set.

### 2.3. Metric Calculation

Both functions now calculate average loss and accuracy by:
1.  Storing per-batch `loss.item()` in a list `batch_losses`.
2.  Storing per-batch accuracy (`correct_predictions_in_batch / batch_size`) in a list `batch_accuracies`.
3.  Computing the mean of these lists using `torch.tensor(list_name).mean().item()` after the loop.
This method is robust and accurately reflects the average of per-batch metrics. 