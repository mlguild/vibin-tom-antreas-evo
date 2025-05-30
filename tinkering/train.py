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
    accelerator,
    num_train_iterations: int,
    current_step: int = 0,
    scheduler: torch.optim.lr_scheduler._LRScheduler = None,
):
    """
    Trains the model for a specified number of iterations using Hugging Face Accelerate.
    """
    model.train()
    batch_losses = []
    batch_accuracies = []

    if num_train_iterations == 0:
        return 0.0, 0.0, current_step

    data_iter = itertools.cycle(dataloader)

    progress_bar_display = accelerator.is_main_process
    progress_bar = tqdm(
        range(num_train_iterations),
        desc=f"Training Iterations ({current_step} to {current_step + num_train_iterations -1})",
        leave=True,
        unit="iter",
        disable=not progress_bar_display,
    )

    for i in progress_bar:
        try:
            inputs, labels = next(data_iter)
        except StopIteration:
            if accelerator.is_main_process:
                console.print(
                    "[warning]Training dataloader iterator unexpectedly exhausted. Breaking."
                )
            break

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        accelerator.backward(loss)
        optimizer.step()

        if scheduler:
            scheduler.step()

        gathered_loss = accelerator.gather(loss.detach()).mean()
        batch_losses.append(gathered_loss.item())

        _, predicted = torch.max(outputs.data, 1)
        batch_correct = (predicted == labels).sum().item()
        gathered_predictions = accelerator.gather(predicted)
        gathered_labels = accelerator.gather(labels)

        correct_across_processes = (
            (gathered_predictions == gathered_labels).sum().item()
        )
        total_on_processes = gathered_labels.size(0)

        batch_accuracy = (
            correct_across_processes / total_on_processes
            if total_on_processes > 0
            else 0.0
        )
        batch_accuracies.append(batch_accuracy)

        if accelerator.is_main_process:
            current_lr = optimizer.param_groups[0]["lr"]
            progress_bar.set_postfix(
                loss=f"{gathered_loss.item():.4f}",
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
    accelerator,
    top_k: tuple = (1, 5),
):
    """
    Validates the model on the entire validation dataset using Hugging Face Accelerate.
    """
    model.eval()
    batch_losses = []
    batch_top_k_correct_counts = {k: [] for k in top_k}
    total_samples_val = 0

    progress_bar_display = accelerator.is_main_process
    progress_bar = tqdm(
        dataloader,
        desc="Validation Pass",
        leave=False,
        unit="batch",
        disable=not progress_bar_display,
    )

    for inputs, labels in progress_bar:
        batch_size_val = inputs.size(0)

        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        gathered_loss_val = accelerator.gather(loss.detach()).mean()
        batch_losses.append(gathered_loss_val.item())

        max_k_val = max(top_k)
        _, pred_top_k = outputs.topk(max_k_val, 1, True, True)
        pred_top_k_gathered = accelerator.gather(pred_top_k.t())
        labels_gathered = accelerator.gather(labels)

        correct_gathered = pred_top_k_gathered.eq(
            labels_gathered.view(1, -1).expand_as(pred_top_k_gathered)
        )

        current_total_on_processes_val = labels_gathered.size(0)

        for k_val in top_k:
            correct_k_gathered = (
                correct_gathered[:k_val]
                .reshape(-1)
                .float()
                .sum(0, keepdim=False)
            )
            batch_top_k_correct_counts[k_val].append(correct_k_gathered.item())

        total_samples_val += current_total_on_processes_val

        if accelerator.is_main_process:
            acc1_batch_disp = (
                (
                    batch_top_k_correct_counts[1][-1]
                    / current_total_on_processes_val
                )
                if 1 in top_k and current_total_on_processes_val > 0
                else 0.0
            )
            postfix_str = f"Loss: {gathered_loss_val.item():.4f}, Acc@1: {acc1_batch_disp:.3f}"
            if (
                5 in top_k
                and 5 in batch_top_k_correct_counts
                and current_total_on_processes_val > 0
            ):
                acc5_batch_disp = (
                    batch_top_k_correct_counts[5][-1]
                    / current_total_on_processes_val
                )
                postfix_str += f", Acc@5: {acc5_batch_disp:.3f}"
            progress_bar.set_postfix_str(postfix_str)

    avg_loss = (
        torch.tensor(batch_losses).mean().item() if batch_losses else 0.0
    )
    avg_top_k_accuracies = {}
    if total_samples_val > 0:
        for k_val in top_k:
            avg_top_k_accuracies[k_val] = (
                sum(batch_top_k_correct_counts[k_val]) / total_samples_val
            )
    else:
        for k_val in top_k:
            avg_top_k_accuracies[k_val] = 0.0

    return avg_loss, avg_top_k_accuracies


if __name__ == "__main__":
    console.print("[info]tinkering/train.py executed directly.")
    console.print(
        "  Contains Accelerate-compatible run_train_iterations and run_validation functions."
    )
