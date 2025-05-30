import torch
import torch.optim as optim
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import fire
from pathlib import Path

from accelerate import Accelerator, DistributedType
from accelerate.utils import set_seed  # For reproducibility

from rich.console import Console
from rich.traceback import install as install_rich_traceback
from rich.table import Table
from rich.progress import Progress
from rich.pretty import pprint

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
    num_workers: int = 4,
    save_path: str = "./saved_models_accelerate",
    model_name: str = "model.pt",
    final_lr_factor: float = 0.01,
    mixed_precision: str = "bf16",
    print_model_summary: bool = True,
    val_top_k: tuple = (1, 5),
    seed: int = 42,
):
    """
    Main training script for image classification using ResNet, with Hugging Face Accelerate.
    """
    install_rich_traceback()
    set_seed(seed)

    # --- Accelerator Setup ---
    accelerator_mixed_precision = mixed_precision.lower()
    if accelerator_mixed_precision not in ["no", "fp16", "bf16"]:
        console.print(
            f"[warning]Invalid mixed_precision '{mixed_precision}'. Defaulting to 'no' (fp32)."
        )
        accelerator_mixed_precision = "no"

    if (
        accelerator_mixed_precision == "bf16"
        and not torch.cuda.is_bf16_supported()
    ):
        console.print(
            "[warning]bf16 requested but not supported on this CUDA device. Falling back to 'no' (fp32) for Accelerator."
        )
        accelerator_mixed_precision = "no"

    if accelerator_mixed_precision == "fp16" and not torch.cuda.is_available():
        console.print(
            "[warning]fp16 requested but no CUDA available. Falling back to 'no' (fp32) for Accelerator."
        )
        accelerator_mixed_precision = "no"

    accelerator = Accelerator(mixed_precision=accelerator_mixed_precision)
    device = accelerator.device

    # Convert val_top_k from list (if from Fire) to tuple
    if isinstance(val_top_k, list):
        val_top_k = tuple(val_top_k)

    console.rule(
        f"[bold green]Starting Training ({accelerator.distributed_type}): {dataset_name.upper()} "
        f"({emnist_split if dataset_name.lower()=='emnist' else ''}) - Precision: {accelerator.mixed_precision.upper() if accelerator.mixed_precision else 'FP32'}[/bold green]"
    )
    console.print(f"[info]Using device: {device}, Seed: {seed}")
    console.print(
        f"[info]Num processes: {accelerator.num_processes}, Process index: {accelerator.process_index}"
    )

    # Specific data directory for the chosen dataset
    dataset_specific_data_dir = Path(data_dir_root) / dataset_name
    if dataset_name.lower() == "emnist":
        dataset_specific_data_dir = (
            Path(data_dir_root) / f"emnist_{emnist_split}"
        )

    console.print(
        f"[info]Data will be loaded/stored in: {dataset_specific_data_dir.resolve()}"
    )

    save_dir = (
        Path(save_path)
        / dataset_name
        / (
            accelerator.mixed_precision
            if accelerator.mixed_precision
            else "fp32"
        )
    )
    # Only main process should create directories and save
    if accelerator.is_main_process:
        save_dir.mkdir(parents=True, exist_ok=True)
        console.print(
            f"[info]Data source: {dataset_specific_data_dir.resolve()}"
        )
        console.print(
            f"[info]Model checkpoints will be saved in: {save_dir.resolve()}"
        )

    final_model_name_stem = f"{dataset_name}_{Path(model_name).stem}_{accelerator.mixed_precision if accelerator.mixed_precision else 'fp32'}"
    model_save_path = (
        save_dir / f"{final_model_name_stem}{Path(model_name).suffix}"
    )
    console.print(f"[info]Model will be saved to: {model_save_path.resolve()}")

    # --- Dataloaders ---
    if accelerator.is_main_process:
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
    if accelerator.is_main_process:
        console.print(
            f"[success]Dataset '{dataset_name}' loaded. Num classes: {num_classes}, Input Channels: {in_channels}"
        )

    # --- Model, Optimizer, Criterion, Scheduler ---
    if accelerator.is_main_process:
        console.print(
            "[info]Initializing model, optimizer, criterion, and LR scheduler..."
        )
    model = ResNet4StageCustom(
        num_classes=num_classes, in_channels=in_channels
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss()

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

    if accelerator.is_main_process:
        console.print(
            "[success]Model, optimizer, criterion, and LR scheduler initialized."
        )
        console.print(f"  Optimizer: AdamW, Initial LR: {lr}")
        console.print(
            f"  LR Scheduler: LambdaLR (linear decay to {final_lr_factor * lr:.2e} over {num_total_iterations} iters)"
        )
        console.print(f"  Criterion: CrossEntropyLoss")

    # --- Prepare with Accelerator ---
    model, optimizer, train_loader, test_loader, scheduler = (
        accelerator.prepare(
            model, optimizer, train_loader, test_loader, scheduler
        )
    )

    if print_model_summary and accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        console.rule("[bold cyan]Model Summary[/bold cyan]")
        console.print(f"Model: {unwrapped_model.__class__.__name__}")
        total_params = sum(p.numel() for p in unwrapped_model.parameters())
        trainable_params = sum(
            p.numel() for p in unwrapped_model.parameters() if p.requires_grad
        )
        console.print(f"Total Parameters: {total_params:,}")
        console.print(f"Trainable Parameters: {trainable_params:,}")
        summary_table = Table(title="Layer Name & Shape")
        summary_table.add_column("Layer Name", style="cyan")
        summary_table.add_column("Parameter Shape", style="magenta")
        summary_table.add_column(
            "Number of Parameters", style="green", justify="right"
        )

        for name, param in unwrapped_model.named_parameters():
            summary_table.add_row(
                name, str(list(param.shape)), f"{param.numel():,}"
            )
        console.print(summary_table)
        # console.print("Full model structure:") # pprint can be very verbose for large models
        # pprint(model)
        console.rule()

    if accelerator.is_main_process:
        console.print("[success]Components prepared with Accelerator.")

    # --- Training Loop ---
    if accelerator.is_main_process:
        console.rule(
            f"[bold blue]Training Started ({accelerator.mixed_precision.upper() if accelerator.mixed_precision else 'FP32'})[/bold blue]"
        )
    current_iteration = 0
    best_val_top1_accuracy = 0.0  # Now specifically tracks Top-1

    overall_progress = None
    if accelerator.is_main_process:
        overall_progress = Progress(console=console, transient=False)
        task_total_iters = overall_progress.add_task(
            "[cyan]Total Iterations", total=num_total_iterations
        )
        overall_progress.start()

    while current_iteration < num_total_iterations:
        iters_to_run = min(
            eval_interval_iters, num_total_iterations - current_iteration
        )

        train_loss, train_acc, updated_iteration = run_train_iterations(
            model,
            train_loader,
            optimizer,
            criterion,
            accelerator,
            num_train_iterations=iters_to_run,
            current_step=current_iteration,
            scheduler=scheduler,
        )
        current_iteration = updated_iteration
        if accelerator.is_main_process and overall_progress:
            overall_progress.update(task_total_iters, advance=iters_to_run)

        if (
            current_iteration % eval_interval_iters == 0
            or current_iteration == num_total_iterations
        ):
            if accelerator.is_main_process:
                console.print(
                    f"Iter: {current_iteration}/{num_total_iterations} | Train Loss: {train_loss:.4f}, Train Acc@1: {train_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}"
                )

            # --- Validation ---
            val_loss, val_top_k_accs = run_validation(
                model,
                test_loader,
                criterion,
                accelerator,
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
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    accelerator.save(
                        unwrapped_model.state_dict(), model_save_path
                    )
                    console.print(
                        f"  Model state_dict saved to {model_save_path}"
                    )
                except Exception as e:
                    console.print(f"[error]Error saving model: {e}")

        if current_iteration >= num_total_iterations:
            break

    if accelerator.is_main_process and overall_progress:
        overall_progress.stop()

    if accelerator.is_main_process:
        console.rule("[bold green]Training Finished[/bold green]")
        console.print(
            f"Model state_dict saved to: {model_save_path} (if best was achieved)"
        )
        console.print(f"Best validation Acc@1: {best_val_top1_accuracy:.4f}")


if __name__ == "__main__":
    fire.Fire(main)
