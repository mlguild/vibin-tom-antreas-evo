import torch
import torch.nn as nn
from tqdm import tqdm
from rich.console import Console
import itertools

console = Console()


def run_train_iterations(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    num_train_iterations: int,
    current_step: int = 0,
    scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    amp_dtype: torch.dtype = None,
):
    """
    Trains the model for a specified number of iterations.

    Args:
        model: The neural network model.
        dataloader: The training DataLoader.
        optimizer: The optimizer.
        criterion: The loss function.
        device: The device to train on (e.g., 'cuda' or 'cpu').
        num_train_iterations: The number of training iterations (steps) to perform.
        current_step: The global step count at the beginning of these iterations.
        scheduler: Optional learning rate scheduler.
        amp_dtype: torch.dtype to use for autocast (e.g., torch.bfloat16, torch.float16), None for fp32.

    Returns:
        tuple: (average_loss, average_accuracy, updated_current_step)
    """
    model.train()
    batch_losses = []
    batch_accuracies = []

    if num_train_iterations == 0:
        return 0.0, 0.0, current_step

    data_iter = itertools.cycle(dataloader)

    use_autocast = amp_dtype is not None and device.type == "cuda"

    progress_bar = tqdm(
        range(num_train_iterations),
        desc=f"Training Iterations ({current_step} to {current_step + num_train_iterations -1})",
        leave=True,  # Keep this bar visible after completion for context
        unit="iter",
    )

    for i in progress_bar:
        try:
            inputs, labels = next(data_iter)
        except (
            StopIteration
        ):  # Should not happen with itertools.cycle but as a safeguard
            console.print(
                "[warning]Training dataloader iterator unexpectedly exhausted despite using itertools.cycle. Breaking."
            )
            break

        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast(
            device_type=device.type,
            enabled=use_autocast,
            dtype=amp_dtype if use_autocast else torch.float32,
        ):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        if scheduler:
            scheduler.step()

        batch_losses.append(loss.item())
        _, predicted = torch.max(outputs.data, 1)
        batch_correct = (predicted == labels).sum().item()
        batch_accuracy = batch_correct / inputs.size(0)
        batch_accuracies.append(batch_accuracy)

        current_lr = optimizer.param_groups[0]["lr"]
        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{batch_accuracy:.3f}",
            lr=f"{current_lr:.1e}",
        )
        current_step += 1

    avg_loss = (
        torch.tensor(batch_losses).mean().item() if batch_losses else 0.0
    )
    avg_accuracy = (
        torch.tensor(batch_accuracies).mean().item()
        if batch_accuracies
        else 0.0
    )
    return avg_loss, avg_accuracy, current_step


def run_validation(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_dtype: torch.dtype = None,
    top_k: tuple = (1, 5),  # Specify which top-k accuracies to compute
):
    """
    Validates the model on the entire validation dataset.

    Args:
        model: The neural network model.
        dataloader: The validation DataLoader.
        criterion: The loss function.
        device: The device to run validation on.
        amp_dtype: torch.dtype to use for autocast (e.g., torch.bfloat16, torch.float16), None for fp32.
        top_k (tuple): A tuple of integers for which top-k accuracies to compute (e.g., (1, 5)).

    Returns:
        tuple: (average_loss, dict_of_avg_top_k_accuracies)
               Example: (0.5, {1: 0.80, 5: 0.95})
    """
    model.eval()
    batch_losses = []
    # Store batch accuracies for each k in top_k
    # e.g., batch_top_k_correct_counts[k] will be a list of correct counts for top-k for each batch
    batch_top_k_correct_counts = {k: [] for k in top_k}
    total_samples = 0

    use_autocast = amp_dtype is not None and device.type == "cuda"

    progress_bar = tqdm(
        dataloader, desc="Validation Pass", leave=False, unit="batch"
    )

    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            batch_size = inputs.size(0)

            with torch.amp.autocast(
                device_type=device.type,
                enabled=use_autocast,
                dtype=amp_dtype if use_autocast else torch.float32,
            ):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            batch_losses.append(loss.item())

            # Calculate Top-k accuracies for this batch
            max_k = max(top_k)
            _, pred_top_k = outputs.topk(max_k, 1, True, True)
            pred_top_k = pred_top_k.t()
            correct = pred_top_k.eq(labels.view(1, -1).expand_as(pred_top_k))

            for k_val in top_k:
                correct_k = (
                    correct[:k_val].reshape(-1).float().sum(0, keepdim=True)
                )
                batch_top_k_correct_counts[k_val].append(correct_k.item())

            total_samples += batch_size

            # Update progress bar postfix (shows Top-1 for brevity)
            acc1_batch = (
                batch_top_k_correct_counts[1][-1] / batch_size
                if 1 in top_k
                else 0.0
            )
            postfix_str = f"Loss: {loss.item():.4f}, Acc@1: {acc1_batch:.3f}"
            if (
                5 in top_k
                and 5 in batch_top_k_correct_counts
                and batch_top_k_correct_counts[5]
            ):
                acc5_batch = batch_top_k_correct_counts[5][-1] / batch_size
                postfix_str += f", Acc@5: {acc5_batch:.3f}"
            progress_bar.set_postfix_str(postfix_str)

    avg_loss = (
        torch.tensor(batch_losses).mean().item() if batch_losses else 0.0
    )

    avg_top_k_accuracies = {}
    if total_samples > 0:
        for k_val in top_k:
            avg_top_k_accuracies[k_val] = (
                sum(batch_top_k_correct_counts[k_val]) / total_samples
            )
    else:
        for k_val in top_k:
            avg_top_k_accuracies[k_val] = 0.0

    return avg_loss, avg_top_k_accuracies


if __name__ == "__main__":
    console.print("[info]tinkering/train.py executed directly.")
    console.print(
        "  Contains run_train_iterations and run_validation functions, now with amp_dtype support."
    )
