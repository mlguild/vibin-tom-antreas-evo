import torch
import torch.nn as nn  # For type hinting if needed, though models are created via function
import fire
from pathlib import Path
import itertools  # For cycling dataloader

from rich.console import Console
from rich.traceback import install as install_rich_traceback
from rich.progress import Progress
from rich.table import Table
from rich.pretty import pprint

from tinkering.datasets import get_dataloaders
from tinkering.models import (
    ResNet4StageCustom,
)  # To define the model structure
from tinkering.evolution import (
    initialize_population,
    run_one_generation,
    get_fitness_evaluation_fn,
    accuracy_metric,
)

console = Console()

# Fitness metric: Negative Cross-Entropy Loss (higher is better for ES)
# Global criterion instance for the fitness metric
_criterion_for_fitness = nn.CrossEntropyLoss()


def negative_loss_fitness_metric(
    outputs: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Calculates negative cross-entropy loss as a fitness metric."""
    loss = _criterion_for_fitness(outputs, labels)
    return -loss  # Higher is better


def main(
    num_generations: int = 100,
    population_size: int = 50,
    mutation_strength: float = 0.05,
    dataset_name: str = "cifar100",
    emnist_split: str = "byclass",
    batch_size: int = 128,  # Batch size for fitness evaluation on each generation
    data_dir_root: str = "./data",
    num_workers: int = 4,
    log_interval_generations: int = 1,
    save_path: str = "./saved_es_models",
    model_name_prefix: str = "model_es_gen",
    dtype_str: str = "bf16",  # Default to bf16
    print_model_summary: bool = True,
    report_top_k_acc: list[int] = (1, 5),
):
    """
    Main script for image classification using a basic Evolutionary Algorithm (truncation selection).
    """
    install_rich_traceback()
    console.rule(
        f"[bold green]Evolutionary Algorithm: {dataset_name.upper()} ({emnist_split if dataset_name.lower()=='emnist' else ''}) - Fitness: Negative Loss, Eval Precision: {dtype_str.upper()}[/bold green]"
    )

    # --- Setup Device and Dtype ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[info]Using device: {device}")

    amp_dtype = None
    actual_dtype_str_used = dtype_str
    if dtype_str.lower() == "bf16":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
            console.print(
                "[info]Using bfloat16 for mixed precision fitness evaluation."
            )
        else:
            console.print(
                "[warning]bfloat16 requested but not supported on this device/PyTorch build. Fitness eval fallback to float32."
            )
            amp_dtype = None
            actual_dtype_str_used = "fp32"
    elif dtype_str.lower() == "fp16":
        amp_dtype = torch.float16
        console.print(
            "[info]Using float16 for mixed precision fitness evaluation."
        )
    elif dtype_str.lower() == "fp32":
        amp_dtype = None
        console.print(
            "[info]Using float32 for fitness evaluation (no mixed precision)."
        )
    else:
        console.print(
            f"[warning]Unknown dtype_str '{dtype_str}'. Defaulting to float32 for fitness evaluation."
        )
        amp_dtype = None
        actual_dtype_str_used = "fp32"

    dataset_specific_data_dir = Path(data_dir_root) / dataset_name
    if dataset_name.lower() == "emnist":
        dataset_specific_data_dir = (
            Path(data_dir_root) / f"emnist_{emnist_split}"
        )
    console.print(f"[info]Data from: {dataset_specific_data_dir.resolve()}")

    save_dir = Path(save_path) / dataset_name / actual_dtype_str_used
    save_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[info]Best models will be saved in: {save_dir.resolve()}")

    # --- Dataloader (for fitness evaluation) ---
    console.print("[info]Loading dataset for fitness evaluation...")
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

    # Using train_loader for fitness evaluation batches, cycling through it
    fitness_eval_loader, _, num_classes, _ = get_dataloaders(
        dataset_name=dataset_name, **dataloader_kwargs
    )
    console.print(
        f"[success]Dataset '{dataset_name}' loaded. Num classes: {num_classes}. Input Channels: {in_channels}"
    )
    fitness_data_iterator = itertools.cycle(fitness_eval_loader)

    # --- Model Creation Function ---
    def model_create_fn():
        return ResNet4StageCustom(
            num_classes=num_classes, in_channels=in_channels
        )

    # Print model summary for the base architecture
    if print_model_summary:
        temp_model_for_summary = model_create_fn()
        console.rule("[bold cyan]Base Model Summary[/bold cyan]")
        console.print(
            f"Model Architecture: {temp_model_for_summary.__class__.__name__}"
        )
        total_params = sum(
            p.numel() for p in temp_model_for_summary.parameters()
        )
        console.print(f"Total Parameters (per individual): {total_params:,}")
        summary_table = Table(title="Layer Name & Shape (for one individual)")
        summary_table.add_column("Layer Name", style="cyan")
        summary_table.add_column("Parameter Shape", style="magenta")
        summary_table.add_column(
            "Number of Parameters", style="green", justify="right"
        )
        for name, param in temp_model_for_summary.named_parameters():
            summary_table.add_row(
                name, str(list(param.shape)), f"{param.numel():,}"
            )
        console.print(summary_table)
        console.rule()
        del temp_model_for_summary

    # --- Initialize Population ---
    console.print(
        f"[info]Initializing population of {population_size} individuals..."
    )
    current_population = initialize_population(
        population_size, model_create_fn, device
    )

    # --- Fitness Evaluation Function ---
    # Create a base model instance on the correct device FOR vmap's architectural reference ONLY.
    # The actual parameters will come from the population during evaluation.
    base_model_for_vmap_setup = model_create_fn().to(device)
    fitness_evaluator = get_fitness_evaluation_fn(
        base_model_for_vmap_setup,
        negative_loss_fitness_metric,
        amp_dtype=amp_dtype,  # Pass determined amp_dtype
    )
    console.print(
        f"[info]Fitness evaluation function (negative loss, using {actual_dtype_str_used.upper()}) prepared."
    )

    # --- Evolutionary Loop ---
    console.rule("[bold blue]Evolution Started[/bold blue]")
    overall_best_fitness = -float("inf")  # Higher is better (negative loss)
    overall_actual_loss_at_best_fitness = float(
        "inf"
    )  # Track actual loss for clarity
    overall_best_top1_acc = 0.0
    overall_best_top5_acc = 0.0
    overall_best_model_state_dict = None

    # Convert report_top_k_acc from list to tuple if it comes from Fire as a list
    if isinstance(report_top_k_acc, list):
        report_top_k_acc = tuple(report_top_k_acc)

    with Progress(console=console, transient=False) as generation_progress:
        task_generations = generation_progress.add_task(
            "[cyan]Generations", total=num_generations
        )
        for gen in range(num_generations):
            # Get a batch of data for this generation's fitness evaluation
            fitness_batch_inputs, fitness_batch_labels = next(
                fitness_data_iterator
            )
            fitness_data_batch_tuple = (
                fitness_batch_inputs.to(device),
                fitness_batch_labels.to(device),
            )

            (
                next_pop,
                best_offspring_sd,
                gen_best_fit,
                gen_avg_fit,
                gen_best_offspring_accs,
            ) = run_one_generation(
                current_population,
                mutation_strength,
                fitness_data_batch_tuple,
                fitness_evaluator,
                accuracy_metric,
                device,
                amp_dtype=amp_dtype,
                top_k_acc_report=report_top_k_acc,
            )
            current_population = next_pop
            generation_progress.update(task_generations, advance=1)

            gen_actual_loss_at_best_fit = (
                -gen_best_fit
            )  # Convert back to actual loss for logging
            gen_actual_avg_loss = (
                -gen_avg_fit
            )  # Convert back to actual loss for logging

            log_msg_parts = [
                f"Gen: {gen+1}/{num_generations}",
                f"BestFit(NegLoss): {gen_best_fit:.4f} (ActualLoss: {gen_actual_loss_at_best_fit:.4f})",
                f"AvgFit(NegLoss): {gen_avg_fit:.4f} (ActualAvgLoss: {gen_actual_avg_loss:.4f})",
            ]
            if 1 in gen_best_offspring_accs:
                log_msg_parts.append(
                    f"BestAcc@1: {gen_best_offspring_accs[1]:.2f}%"
                )
            if 5 in gen_best_offspring_accs:
                log_msg_parts.append(
                    f"BestAcc@5: {gen_best_offspring_accs[5]:.2f}%"
                )

            if (gen + 1) % log_interval_generations == 0:
                console.print(" | ".join(log_msg_parts))

            if gen_best_fit > overall_best_fitness:
                overall_best_fitness = gen_best_fit
                overall_actual_loss_at_best_fitness = (
                    -overall_best_fitness
                )  # Store the actual loss
                overall_best_model_state_dict = best_offspring_sd
                # Update overall best accuracies from the current best offspring
                overall_best_top1_acc = gen_best_offspring_accs.get(1, 0.0)
                overall_best_top5_acc = gen_best_offspring_accs.get(5, 0.0)

                fname = f"{model_name_prefix}_{dataset_name}_gen_{gen+1}_fit_{overall_best_fitness:.4f}_loss_{overall_actual_loss_at_best_fitness:.4f}_acc1_{overall_best_top1_acc:.2f}_{actual_dtype_str_used}.pt"
                model_save_path = save_dir / fname
                console.print(
                    f"[bold green]New best neg_loss: {overall_best_fitness:.4f} (Actual Loss: {overall_actual_loss_at_best_fitness:.4f}, Acc@1: {overall_best_top1_acc:.2f}%) at Gen {gen+1}. Saving...[/bold green]"
                )
                if overall_best_model_state_dict:
                    torch.save(overall_best_model_state_dict, model_save_path)

    console.rule("[bold green]Evolution Finished[/bold green]")
    if overall_best_model_state_dict:
        console.print(
            f"Overall best fitness (neg_loss): {overall_best_fitness:.4f} (Actual Loss: {overall_actual_loss_at_best_fitness:.4f})"
        )
        console.print(
            f"  Corresp. Acc@1: {overall_best_top1_acc:.2f}%, Acc@5: {overall_best_top5_acc:.2f}%"
        )
        final_fname = f"{model_name_prefix}_{dataset_name}_FINAL_fit_{overall_best_fitness:.4f}_loss_{overall_actual_loss_at_best_fitness:.4f}_acc1_{overall_best_top1_acc:.2f}_{actual_dtype_str_used}.pt"
        final_save_path = save_dir / final_fname
        torch.save(overall_best_model_state_dict, final_save_path)
        console.print(f"Final best model saved to: {final_save_path}")
    else:
        console.print("No best model saved.")


if __name__ == "__main__":
    fire.Fire(main)
