import torch
import torch.nn as nn
import fire
from pathlib import Path
import itertools
import copy
import time

from accelerate import Accelerator, DistributedType
from accelerate.utils import set_seed

from rich.console import Console
from rich.traceback import install as install_rich_traceback
from rich.progress import Progress
from rich.table import Table
from rich.pretty import pprint

from tinkering.datasets import get_dataloaders
from tinkering.models import ResNet4StageCustom
from tinkering.evolution import (
    PopulationOptimizer,
    get_fitness_evaluation_fn,
    negative_loss_fitness_metric,
    accuracy_metric,
)

console = Console()


def main(
    num_generations: int = 100,
    population_size: int = 50,
    mutation_strength: float = 0.05,
    dataset_name: str = "cifar100",
    emnist_split: str = "byclass",
    batch_size: int = 128,
    data_dir_root: str = "./data",
    num_workers: int = 4,
    log_interval_generations: int = 1,
    save_path: str = "./saved_pop_opt_models",
    model_name_prefix: str = "model_pop_opt_gen",
    mixed_precision: str = "bf16",
    print_model_summary: bool = True,
    report_top_k_acc: tuple = (1, 5),
    seed: int = 42,
):
    """
    Main script for image classification using PopulationOptimizer (mutation & truncation selection).
    """
    install_rich_traceback()
    set_seed(seed)

    accelerator = Accelerator(mixed_precision=mixed_precision.lower())
    device = accelerator.device

    amp_dtype_for_fitness_eval = None
    actual_mp_used_for_fitness = (
        accelerator.mixed_precision if accelerator.mixed_precision else "fp32"
    )
    if accelerator.mixed_precision == "bf16":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            amp_dtype_for_fitness_eval = torch.bfloat16
        else:
            if accelerator.is_main_process:
                console.print(
                    "[warning]bf16 requested but not supported. Fitness eval using fp32."
                )
            actual_mp_used_for_fitness = "fp32"
            amp_dtype_for_fitness_eval = None
    elif accelerator.mixed_precision == "fp16":
        amp_dtype_for_fitness_eval = torch.float16
    elif (
        accelerator.mixed_precision == "no"
        or accelerator.mixed_precision is None
    ):
        amp_dtype_for_fitness_eval = None
        actual_mp_used_for_fitness = "fp32"

    if isinstance(report_top_k_acc, list):
        report_top_k_acc = tuple(report_top_k_acc)

    if accelerator.is_main_process:
        console.rule(
            f"[bold green]PopulationOptimizer Training ({accelerator.distributed_type}): {dataset_name.upper()} "
            f"({emnist_split if dataset_name.lower()=='emnist' else ''}) - Eval Precision: {actual_mp_used_for_fitness.upper()}[/bold green]"
        )
        console.print(
            f"[info]Device: {device}, Seed: {seed}, Pop Size: {population_size}, Mut Strength: {mutation_strength}"
        )

    dataset_specific_data_dir = Path(data_dir_root) / dataset_name
    if dataset_name.lower() == "emnist":
        dataset_specific_data_dir = (
            Path(data_dir_root) / f"emnist_{emnist_split}"
        )
    console.print(f"[info]Data from: {dataset_specific_data_dir.resolve()}")

    save_dir = Path(save_path) / dataset_name / actual_mp_used_for_fitness
    if accelerator.is_main_process:
        save_dir.mkdir(parents=True, exist_ok=True)

    # --- Dataloaders ---
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
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    fitness_eval_loader, _, num_classes, _ = get_dataloaders(
        dataset_name=dataset_name, **dataloader_kwargs
    )
    # Prepare only the dataloader with accelerator for proper sharding if distributed
    fitness_eval_loader = accelerator.prepare(fitness_eval_loader)
    fitness_data_iterator = itertools.cycle(fitness_eval_loader)

    # --- Model Creation Function & Optimizer Setup ---
    def model_create_fn():
        # Models created by optimizer will be moved to its device internally
        return ResNet4StageCustom(
            num_classes=num_classes, in_channels=in_channels
        )

    if print_model_summary and accelerator.is_main_process:
        temp_model_for_summary = model_create_fn()
        console.rule("[bold cyan]Base Model Summary[/bold cyan]")
        console.print(
            f"Model Architecture: {temp_model_for_summary.__class__.__name__}"
        )
        total_params = sum(
            p.numel() for p in temp_model_for_summary.parameters()
        )
        console.print(f"Total Parameters (per individual): {total_params:,}")
        summary_table = Table(title="Layer Name & Shape")
        summary_table.add_column("Layer Name", style="cyan")
        summary_table.add_column("Parameter Shape", style="magenta")
        summary_table.add_column(
            "Number of Parameters", style="green", justify="right"
        )
        for name, param in temp_model_for_summary.named_parameters():
            if param.requires_grad:
                summary_table.add_row(
                    name, str(list(param.shape)), f"{param.numel():,}"
                )
        console.print(summary_table)
        console.rule()
        del temp_model_for_summary

    pop_optimizer = PopulationOptimizer(
        population_size=population_size,
        model_create_fn=model_create_fn,
        mutation_strength=mutation_strength,
        device=device,
    )

    # Fitness evaluator uses the base_model_architecture_template from the optimizer
    fitness_evaluator = get_fitness_evaluation_fn(
        pop_optimizer.base_model_architecture_template,
        negative_loss_fitness_metric,
        amp_dtype=amp_dtype_for_fitness_eval,
    )
    if accelerator.is_main_process:
        console.print(
            f"[info]PopulationOptimizer and Fitness Evaluator prepared."
        )

    # --- Evolutionary Loop ---
    if accelerator.is_main_process:
        console.rule("[bold blue]Evolution Started[/bold blue]")
    overall_best_fitness = -float("inf")
    overall_actual_loss_at_best_fitness = float("inf")
    overall_best_top1_acc = 0.0
    overall_best_top5_acc = 0.0
    overall_best_model_state_dict = None

    overall_progress = None
    if accelerator.is_main_process:
        overall_progress = Progress(console=console, transient=False)
        task_generations = overall_progress.add_task(
            "[cyan]Generations", total=num_generations
        )
        overall_progress.start()

    for gen in range(num_generations):
        gen_time_start = time.time()
        # 1. Generate offspring population for evaluation
        offspring_to_evaluate = (
            pop_optimizer.generate_offspring_for_evaluation()
        )

        # 2. Evaluate these offspring to get fitness scores
        fitness_batch_inputs, fitness_batch_labels = next(
            fitness_data_iterator
        )
        fitness_data_batch_tuple = (fitness_batch_inputs, fitness_batch_labels)

        fitness_scores = fitness_evaluator(
            offspring_to_evaluate, fitness_data_batch_tuple, device
        )

        # 3. Update the optimizer's internal population based on fitness
        pop_optimizer.update_population_with_selected_offspring(
            offspring_to_evaluate, fitness_scores
        )

        # 4. Log and get the best individual from the *evaluated offspring* for this generation
        best_offspring_instance_this_gen, gen_best_fit = (
            pop_optimizer.get_best_individual_from_evaluated_offspring(
                offspring_to_evaluate, fitness_scores
            )
        )
        gen_avg_fit = (
            fitness_scores.mean().item()
            if fitness_scores.numel() > 0
            else -float("inf")
        )
        gen_best_offspring_accs = {k: 0.0 for k in report_top_k_acc}

        if best_offspring_instance_this_gen:
            # Evaluate accuracy of the single best offspring model from this generation
            inputs_acc, labels_acc = fitness_data_batch_tuple
            use_autocast_acc = (
                amp_dtype_for_fitness_eval is not None
                and device.type == "cuda"
            )
            with (
                torch.no_grad(),
                torch.amp.autocast(
                    device_type=device.type,
                    enabled=use_autocast_acc,
                    dtype=(
                        amp_dtype_for_fitness_eval
                        if use_autocast_acc
                        else torch.float32
                    ),
                ),
            ):
                outputs_best = best_offspring_instance_this_gen(inputs_acc)
            acc_dict_tensors = accuracy_metric(
                outputs_best, labels_acc, top_k=report_top_k_acc
            )
            gen_best_offspring_accs = {
                k: v.item() for k, v in acc_dict_tensors.items()
            }

        if accelerator.is_main_process and overall_progress:
            overall_progress.update(task_generations, advance=1)

        if (
            accelerator.is_main_process
            and (gen + 1) % log_interval_generations == 0
        ):
            gen_actual_loss_at_best_fit = (
                -gen_best_fit
                if gen_best_fit != -float("inf")
                else float("inf")
            )
            gen_actual_avg_loss = (
                -gen_avg_fit if gen_avg_fit != -float("inf") else float("inf")
            )
            log_msg_parts = [
                f"Gen: {gen+1}/{num_generations}",
                f"BestFit(NegLoss): {gen_best_fit:.4f} (ActualLoss: {gen_actual_loss_at_best_fit:.4f})",
                f"AvgFit(NegLoss): {gen_avg_fit:.4f} (ActualAvgLoss: {gen_actual_avg_loss:.4f})",
            ]
            if 1 in gen_best_offspring_accs:
                log_msg_parts.append(
                    f"BestAcc@1: {gen_best_offspring_accs.get(1, 0):.2f}%"
                )
            if 5 in gen_best_offspring_accs:
                log_msg_parts.append(
                    f"BestAcc@5: {gen_best_offspring_accs.get(5, 0):.2f}%"
                )
            log_msg_parts.append(f"Time: {time.time()-gen_time_start:.2f}s")
            console.print(" | ".join(log_msg_parts))

        if (
            accelerator.is_main_process
            and best_offspring_instance_this_gen
            and gen_best_fit > overall_best_fitness
        ):
            overall_best_fitness = gen_best_fit
            overall_actual_loss_at_best_fitness = -overall_best_fitness
            overall_best_top1_acc = gen_best_offspring_accs.get(1, 0.0)
            overall_best_top5_acc = gen_best_offspring_accs.get(5, 0.0)
            overall_best_model_state_dict = copy.deepcopy(
                best_offspring_instance_this_gen.cpu().state_dict()
            )

            fname = f"{model_name_prefix}_{dataset_name}_gen_{gen+1}_fit_{overall_best_fitness:.4f}_loss_{overall_actual_loss_at_best_fitness:.4f}_acc1_{overall_best_top1_acc:.2f}_{actual_mp_used_for_fitness}.pt"
            model_save_path = save_dir / fname
            console.print(
                f"[bold green]New overall best neg_loss: {overall_best_fitness:.4f} (Actual Loss: {overall_actual_loss_at_best_fitness:.4f}, Acc@1: {overall_best_top1_acc:.2f}%) at Gen {gen+1}. Saving...[/bold green]"
            )
            if overall_best_model_state_dict:
                accelerator.save(
                    overall_best_model_state_dict, model_save_path
                )
                console.print(f"  Saved to {model_save_path}")

    if accelerator.is_main_process and overall_progress:
        overall_progress.stop()

    if accelerator.is_main_process:
        console.rule("[bold green]Evolution Finished[/bold green]")
        if overall_best_model_state_dict:
            console.print(
                f"Overall best fitness (neg_loss): {overall_best_fitness:.4f} (Actual Loss: {overall_actual_loss_at_best_fitness:.4f})"
            )
            console.print(
                f"  Corresp. Acc@1: {overall_best_top1_acc:.2f}%, Acc@5: {overall_best_top5_acc:.2f}%"
            )
            final_fname = f"{model_name_prefix}_{dataset_name}_FINAL_fit_{overall_best_fitness:.4f}_loss_{overall_actual_loss_at_best_fitness:.4f}_acc1_{overall_best_top1_acc:.2f}_{actual_mp_used_for_fitness}.pt"
            final_save_path = save_dir / final_fname
            accelerator.save(overall_best_model_state_dict, final_save_path)
            console.print(f"Final best model saved to: {final_save_path}")
        else:
            console.print("No best model saved.")


if __name__ == "__main__":
    fire.Fire(main)
