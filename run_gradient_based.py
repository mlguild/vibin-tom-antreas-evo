import torch
import torch.optim as optim
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import fire
from pathlib import Path

from rich.console import Console
from rich.traceback import install as install_rich_traceback
from rich.table import Table
from rich.progress import Progress  # For overall progress
from rich.pretty import pprint  # For model summary

from tinkering.datasets import get_dataloaders
from tinkering.models import ResNet4StageCustom
from tinkering.train import run_train_iterations, run_validation

console = Console()


def main(
    num_total_iterations: int = 20000,
    lr: float = 1e-5,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    dataset_name: str = "cifar100",
    emnist_split: str = "byclass",
    data_dir_root: str = "./data",
    eval_interval_iters: int = 500,
    log_interval_iters: int = 100,
    num_workers: int = 4,
    save_path: str = "./saved_models",
    model_name: str = "model.pt",
    final_lr_factor: float = 0.01,
    dtype_str: str = "bf16",
    print_model_summary: bool = True,  # Added flag for model summary
    val_top_k: list[int] = (1, 5),  # CLI arg for top_k values
):
    """
    Main training script for image classification using ResNet.
    """
    install_rich_traceback()
    console.rule(
        f"[bold green]Starting Training: {dataset_name.upper()} ({emnist_split if dataset_name.lower()=='emnist' else ''}) - Precision: {dtype_str.upper()}[/bold green]"
    )

    # --- Setup Device and Dtype ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[info]Using device: {device}")

    amp_dtype = None
    actual_dtype_str_used = dtype_str
    if dtype_str.lower() == "bf16":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
            console.print("[info]Using bfloat16 for mixed precision.")
        else:
            console.print(
                "[warning]bfloat16 requested but not supported on this device/PyTorch build. Falling back to float32."
            )
            amp_dtype = None  # Fallback to float32
            actual_dtype_str_used = "fp32"
    elif dtype_str.lower() == "fp16":
        amp_dtype = torch.float16
        console.print(
            "[info]Using float16 for mixed precision. Consider GradScaler for stability if issues arise."
        )
    elif dtype_str.lower() != "fp32":
        console.print(
            f"[warning]Unknown dtype '{dtype_str}'. Defaulting to float32."
        )
        amp_dtype = None
        actual_dtype_str_used = "fp32"

    # Specific data directory for the chosen dataset
    dataset_specific_data_dir = Path(data_dir_root) / dataset_name
    if dataset_name.lower() == "emnist":
        dataset_specific_data_dir = (
            Path(data_dir_root) / f"emnist_{emnist_split}"
        )

    console.print(
        f"[info]Data will be loaded/stored in: {dataset_specific_data_dir.resolve()}"
    )

    save_dir = Path(save_path) / dataset_name / actual_dtype_str_used
    save_dir.mkdir(parents=True, exist_ok=True)
    final_model_name = f"{dataset_name}_{Path(model_name).stem}_{actual_dtype_str_used}{Path(model_name).suffix}"
    model_save_path = save_dir / final_model_name
    console.print(f"[info]Model will be saved to: {model_save_path.resolve()}")

    # --- Dataloaders ---
    console.print("[info]Loading datasets...")
    dataloader_kwargs = {
        "data_dir": str(dataset_specific_data_dir),
        "batch_size": batch_size,
        "num_workers": num_workers,
    }
    if dataset_name.lower() == "emnist":
        dataloader_kwargs["emnist_split"] = emnist_split
        in_channels = 1
    elif dataset_name.lower() == "cifar100":
        in_channels = 3
    else:
        raise ValueError(
            f"Unsupported dataset_name for input channels: {dataset_name}"
        )

    train_loader, test_loader, num_classes, _ = get_dataloaders(
        dataset_name=dataset_name, **dataloader_kwargs
    )
    console.print(
        f"[success]Dataset '{dataset_name}' loaded. Num classes: {num_classes}, Input Channels: {in_channels}"
    )

    # --- Model, Optimizer, Criterion ---
    console.print(
        "[info]Initializing model, optimizer, criterion, and LR scheduler..."
    )
    model = ResNet4StageCustom(
        num_classes=num_classes, in_channels=in_channels
    ).to(device)

    if print_model_summary:
        console.rule("[bold cyan]Model Summary[/bold cyan]")
        console.print(f"Model: {model.__class__.__name__}")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        console.print(f"Total Parameters: {total_params:,}")
        console.print(f"Trainable Parameters: {trainable_params:,}")
        summary_table = Table(title="Layer Name & Shape")
        summary_table.add_column("Layer Name", style="cyan")
        summary_table.add_column("Parameter Shape", style="magenta")
        summary_table.add_column(
            "Number of Parameters", style="green", justify="right"
        )

        for name, param in model.named_parameters():
            summary_table.add_row(
                name, str(list(param.shape)), f"{param.numel():,}"
            )
        console.print(summary_table)
        # console.print("Full model structure:") # pprint can be very verbose for large models
        # pprint(model)
        console.rule()

    optimizer = optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    # --- Learning Rate Scheduler ---
    # Linear decay from 1.0 to final_lr_factor over num_total_iterations
    # Ensure lambda function handles iteration counts correctly (0 to num_total_iterations-1)
    def lr_lambda_fn(current_iter):
        if (
            num_total_iterations <= 1
        ):  # Avoid division by zero or weird behavior for 1 iter
            return 1.0
        return 1.0 - (1.0 - final_lr_factor) * (
            current_iter / (num_total_iterations - 1)
        )

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda_fn)

    console.print(
        "[success]Model, optimizer, criterion, and LR scheduler initialized."
    )
    console.print(f"  Optimizer: AdamW, Initial LR: {lr}")
    console.print(
        f"  LR Scheduler: LambdaLR (linear decay to {final_lr_factor * lr:.2e} over {num_total_iterations} iters)"
    )
    console.print(f"  Criterion: CrossEntropyLoss")

    # --- Training Loop ---
    console.rule(
        f"[bold blue]Training Started ({actual_dtype_str_used.upper()})[/bold blue]"
    )
    current_iteration = 0
    best_val_top1_accuracy = 0.0  # Now specifically tracks Top-1

    with Progress(console=console, transient=True) as overall_progress:
        task_total_iters = overall_progress.add_task(
            "[cyan]Total Iterations", total=num_total_iterations
        )

        while current_iteration < num_total_iterations:
            iters_to_run = min(
                eval_interval_iters, num_total_iterations - current_iteration
            )

            train_loss, train_acc, updated_iteration = run_train_iterations(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                num_train_iterations=iters_to_run,
                current_step=current_iteration,
                scheduler=scheduler,
                amp_dtype=amp_dtype,
            )
            current_iteration = updated_iteration
            overall_progress.update(task_total_iters, advance=iters_to_run)

            if (
                current_iteration % log_interval_iters < iters_to_run
            ):  # Log if it falls within the last segment of iters_to_run
                if (
                    current_iteration % eval_interval_iters == 0
                    or current_iteration == num_total_iterations
                ):
                    console.print(
                        f"Iter: {current_iteration}/{num_total_iterations} | Train Loss: {train_loss:.4f}, Train Acc@1: {train_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}"
                    )

            # --- Validation ---
            val_loss, val_top_k_accs = run_validation(
                model,
                test_loader,
                criterion,
                device,
                amp_dtype=amp_dtype,
                top_k=val_top_k,
            )

            val_log_parts = [f"Val Loss: {val_loss:.4f}"]
            if 1 in val_top_k_accs:
                val_log_parts.append(f"Acc@1: {val_top_k_accs[1]:.4f}")
            if 5 in val_top_k_accs:
                val_log_parts.append(f"Acc@5: {val_top_k_accs[5]:.4f}")
            console.print(
                f"[bold yellow]Iter: {current_iteration}/{num_total_iterations} | {' | '.join(val_log_parts)}[/bold yellow]"
            )

            current_val_top1 = val_top_k_accs.get(
                1, 0.0
            )  # Get Top-1 for saving best model
            if current_val_top1 > best_val_top1_accuracy:
                best_val_top1_accuracy = current_val_top1
                console.print(
                    f"[bold green]New best val Acc@1: {best_val_top1_accuracy:.4f}. Saving model to {model_save_path}...[/bold green]"
                )
                try:
                    torch.save(model.state_dict(), model_save_path)
                except Exception as e:
                    console.print(f"[error]Error saving model: {e}")

            if current_iteration >= num_total_iterations:
                break

    console.rule("[bold green]Training Finished[/bold green]")
    console.print(f"Final model saved to: {model_save_path}")
    console.print(f"Best validation Acc@1: {best_val_top1_accuracy:.4f}")


if __name__ == "__main__":
    fire.Fire(main)
