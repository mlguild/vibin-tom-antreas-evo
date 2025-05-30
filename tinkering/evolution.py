import torch
import torch.nn as nn
from torch.func import functional_call, vmap, stack_module_state
import copy
import random

from rich.console import Console

console = Console()

# --- Fitness & Accuracy Metrics ---
_criterion_for_loss_metric = (
    nn.CrossEntropyLoss()
)  # Used if fitness is loss-based


def negative_loss_fitness_metric(
    outputs: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Calculates negative cross-entropy loss. Higher is better."""
    loss = _criterion_for_loss_metric(outputs, labels)
    return -loss


def accuracy_metric(
    outputs: torch.Tensor, labels: torch.Tensor, top_k: tuple = (1,)
) -> dict[int, torch.Tensor]:
    """Calculates top-k accuracy. Returns a dict {k: accuracy_tensor}."""
    max_k = max(top_k)
    batch_size = labels.size(0)

    _, pred = outputs.topk(max_k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(labels.view(1, -1).expand_as(pred))

    res = {}
    for k_val in top_k:
        correct_k = correct[:k_val].reshape(-1).float().sum(0, keepdim=True)
        res[k_val] = correct_k.mul_(
            100.0 / batch_size
        )  # Accuracy as percentage
    return res


def initialize_population(
    population_size: int, model_create_fn, device: torch.device
) -> list[nn.Module]:
    """
    Initializes a population of model instances with random weights.

    Args:
        population_size: The number of individuals in the population.
        model_create_fn: A callable that returns a new model instance (e.g., lambda: MyModelClass(args)).
        device: The torch device to move models to.

    Returns:
        A list of model instances.
    """
    population = []
    for _ in range(population_size):
        model = model_create_fn().to(device)
        # Models are already randomly initialized by PyTorch layers by default
        population.append(model)
    console.print(
        f"[info]Initialized population with {population_size} individuals on {device}."
    )
    return population


def apply_mutation(
    population_models: list[nn.Module],
    mutation_strength: float,
    device: torch.device,
) -> list[nn.Module]:
    """
    Applies Gaussian noise mutation to a population of models.
    Creates new instances for the mutated offspring.

    Args:
        population_models: A list of parent nn.Module instances.
        mutation_strength: Standard deviation of the Gaussian noise to add (sigma).
        device: The torch device.

    Returns:
        A new list of mutated nn.Module instances (offspring).
    """
    offspring_population = []
    for parent_model in population_models:
        offspring_model = copy.deepcopy(parent_model).to(device)
        with torch.no_grad():
            for param in offspring_model.parameters():
                noise = (
                    torch.randn_like(param, device=device) * mutation_strength
                )
                param.add_(noise)
        offspring_population.append(offspring_model)
    # console.print(f"[debug]Applied mutation with strength {mutation_strength} to {len(population_models)} individuals.")
    return offspring_population


def get_fitness_evaluation_fn(
    base_model_for_vmap: nn.Module,
    fitness_metric_fn,
    amp_dtype: torch.dtype = None,  # Added for autocast type
):
    """
    Returns a function that evaluates a population of models on a single data batch.

    Args:
        base_model_for_vmap: A template nn.Module instance (defines architecture for functional_call).
        fitness_metric_fn: A function(outputs, labels) -> scalar_fitness.
        amp_dtype: torch.dtype to use for autocast, None for fp32.

    Returns:
        A function evaluate_population_on_batch(population_models_list, data_batch_tuple, device) -> torch.Tensor (fitness scores)
    """

    def _functional_forward_core_eval(model_params_and_buffers, x_batch):
        return functional_call(
            base_model_for_vmap, model_params_and_buffers, (x_batch,)
        )

    vmapped_get_outputs = vmap(
        _functional_forward_core_eval, in_dims=(0, None), randomness="error"
    )

    def evaluate_population_on_batch(
        population_models_list: list[nn.Module],
        data_batch_tuple: tuple,
        device: torch.device,  # device is now passed here
    ) -> torch.Tensor:
        if not population_models_list:
            return torch.empty(0, device=device)

        inputs, labels = data_batch_tuple
        inputs, labels = inputs.to(device), labels.to(device)

        stacked_params, stacked_buffers = stack_module_state(
            population_models_list
        )
        params_and_buffers_for_vmap = (stacked_params, stacked_buffers)

        use_autocast = amp_dtype is not None and device.type == "cuda"

        with torch.no_grad():
            with torch.amp.autocast(
                device_type=device.type,
                enabled=use_autocast,
                dtype=amp_dtype if use_autocast else torch.float32,
            ):
                batched_outputs = vmapped_get_outputs(
                    params_and_buffers_for_vmap, inputs
                )

        fitness_scores = []
        for i in range(len(population_models_list)):
            outputs_for_model_i = batched_outputs[i]
            fitness = fitness_metric_fn(outputs_for_model_i, labels)
            fitness_scores.append(
                fitness.item() if hasattr(fitness, "item") else fitness
            )

        return torch.tensor(fitness_scores, device=device)

    return evaluate_population_on_batch


def select_top_n_offspring(
    offspring_population: list[nn.Module],
    fitness_scores: torch.Tensor,
    num_to_select: int,
) -> list[nn.Module]:
    """
    Selects the top N individuals from the offspring population based on fitness scores.

    Args:
        offspring_population: The list of mutated nn.Module instances (offspring).
        fitness_scores: A tensor of fitness scores (higher is better) corresponding to offspring_population.
        num_to_select: The number of individuals to select for the next generation.

    Returns:
        A new list of selected nn.Module instances (deep copies of the best offspring).
    """
    if not offspring_population or fitness_scores.numel() == 0:
        return []

    # Ensure num_to_select is not greater than the population size
    num_to_select = min(num_to_select, len(offspring_population))

    # Sort by fitness scores in descending order (higher is better)
    sorted_indices = torch.argsort(fitness_scores, descending=True)

    selected_population = []
    for i in range(num_to_select):
        winner_idx = sorted_indices[i].item()
        # Winners are deepcopied to ensure the new generation has distinct model objects
        selected_population.append(
            copy.deepcopy(offspring_population[winner_idx])
        )

    # console.print(f"[debug]Selected top {len(selected_population)} individuals.")
    return selected_population


def run_one_generation(
    current_population_models: list[nn.Module],
    base_model_architecture: nn.Module,
    mutation_strength: float,
    data_batch_tuple: tuple,
    fitness_eval_fn,
    accuracy_eval_fn,
    device: torch.device,
    amp_dtype: torch.dtype = None,
    top_k_acc_report: tuple = (1, 5),
):
    """
    Runs one generation: mutate, evaluate fitness (neg loss), select.
    Additionally, evaluates accuracy of the best offspring using functional_call.
    """
    offspring_population = apply_mutation(
        current_population_models, mutation_strength, device
    )

    offspring_fitness_scores = fitness_eval_fn(
        offspring_population, data_batch_tuple, device
    )

    best_fitness_this_gen = -float("inf")
    avg_fitness_this_gen = -float("inf")
    best_offspring_state_dict = None
    best_offspring_top_k_accuracies = {k: 0.0 for k in top_k_acc_report}

    if offspring_fitness_scores.numel() > 0:
        best_fitness_this_gen = offspring_fitness_scores.max().item()
        avg_fitness_this_gen = offspring_fitness_scores.mean().item()
        best_offspring_idx = offspring_fitness_scores.argmax().item()

        best_offspring_model_instance = offspring_population[
            best_offspring_idx
        ]
        best_offspring_state_dict = copy.deepcopy(
            best_offspring_model_instance.cpu().state_dict()
        )

        # Prepare params and buffers for the single best model for functional_call
        current_best_params = {
            k: v.to(device)
            for k, v in best_offspring_model_instance.named_parameters()
        }
        current_best_buffers = {
            k: v.to(device)
            for k, v in best_offspring_model_instance.named_buffers()
        }
        best_model_params_and_buffers = (
            current_best_params,
            current_best_buffers,
        )

        inputs_acc, labels_acc = data_batch_tuple
        inputs_acc, labels_acc = inputs_acc.to(device), labels_acc.to(device)
        use_autocast_acc = amp_dtype is not None and device.type == "cuda"
        with torch.no_grad(), torch.amp.autocast(
            device_type=device.type,
            enabled=use_autocast_acc,
            dtype=amp_dtype if use_autocast_acc else torch.float32,
        ):
            outputs_best_offspring = functional_call(
                base_model_architecture,
                best_model_params_and_buffers,
                (inputs_acc,),
            )

        acc_dict_tensors = accuracy_eval_fn(
            outputs_best_offspring, labels_acc, top_k=top_k_acc_report
        )
        best_offspring_top_k_accuracies = {
            k: v.item() for k, v in acc_dict_tensors.items()
        }

    num_to_select_for_next_gen = len(current_population_models)
    next_generation_population = select_top_n_offspring(
        offspring_population,
        offspring_fitness_scores,
        num_to_select_for_next_gen,
    )

    return (
        next_generation_population,
        best_offspring_state_dict,
        best_fitness_this_gen,
        avg_fitness_this_gen,
        best_offspring_top_k_accuracies,
    )


if __name__ == "__main__":
    from tinkering.models import ResNet4StageCustom  # For testing
    from rich.traceback import install as install_rich_traceback

    install_rich_traceback()

    console.rule(
        "[bold green]Testing Evolutionary Components (Functional Acc Eval)[/bold green]"
    )

    # --- Setup for Test ---
    pop_size_test = 4
    num_classes_test = 10
    mutation_strength_test = 0.1
    device_test = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _model_fn():
        return ResNet4StageCustom(num_classes=num_classes_test, in_channels=1)

    # Using negative loss as fitness (higher is better)
    criterion_test = nn.CrossEntropyLoss()

    def _negative_loss_metric(outputs, labels):
        loss = criterion_test(outputs, labels)
        return -loss  # Negative loss, so higher is better

    console.print(
        f"Test params: pop_size={pop_size_test}, mutation_strength={mutation_strength_test}, fitness=negative_loss, device={device_test}"
    )

    # 1. Initialize Population
    console.rule("1. Initialize Population")
    current_pop = initialize_population(pop_size_test, _model_fn, device_test)
    console.print(f"Initialized population of {len(current_pop)} models.")

    # 2. Get Fitness Evaluation Function
    console.rule("2. Prepare Fitness Evaluation")
    base_model_arch_instance = _model_fn().to(device_test)
    fitness_evaluator = get_fitness_evaluation_fn(
        base_model_arch_instance, _negative_loss_metric, amp_dtype=None
    )
    console.print("Fitness evaluator (negative loss based) created.")

    dummy_inputs = torch.randn(16, 1, 28, 28, device=device_test)
    dummy_labels = torch.randint(
        0, num_classes_test, (16,), device=device_test
    )
    dummy_batch = (dummy_inputs, dummy_labels)
    console.print(
        f"Dummy data batch: inputs {dummy_inputs.shape}, labels {dummy_labels.shape}"
    )

    # 3. Run One Generation
    console.rule("3. Run One Generation")
    next_pop, best_state_dict, best_fit, avg_fit, best_accs = (
        run_one_generation(
            current_population_models=current_pop,
            base_model_architecture=base_model_arch_instance,
            mutation_strength=mutation_strength_test,
            data_batch_tuple=dummy_batch,
            fitness_eval_fn=fitness_evaluator,
            accuracy_eval_fn=accuracy_metric,
            device=device_test,
            amp_dtype=None,
            top_k_acc_report=(1, 5),
        )
    )
    console.print(f"Next generation created with {len(next_pop)} individuals.")
    console.print(
        f"Generation - Best Fitness (Neg Loss): {best_fit:.4f}, Avg Fitness: {avg_fit:.4f}"
    )
    if best_state_dict:
        console.print(
            f"  Best model state dict (gen 1) keys: {list(best_state_dict.keys())[:3]}"
        )
    console.print(
        f"Best Offspring Accuracies: Top-1: {best_accs.get(1,0):.2f}%, Top-5: {best_accs.get(5,0):.2f}%"
    )

    # --- Test a second generation ---
    console.rule("4. Run Second Generation (sanity check)")
    current_pop = next_pop
    fitness_evaluator_test_gen2 = get_fitness_evaluation_fn(
        base_model_arch_instance, _negative_loss_metric, amp_dtype=None
    )
    next_pop_2, best_state_dict_2, best_fit_2, avg_fit_2, best_accs_2 = (
        run_one_generation(
            current_population_models=current_pop,
            base_model_architecture=base_model_arch_instance,
            mutation_strength=mutation_strength_test,
            data_batch_tuple=dummy_batch,
            fitness_eval_fn=fitness_evaluator_test_gen2,
            accuracy_eval_fn=accuracy_metric,
            device=device_test,
            amp_dtype=None,
            top_k_acc_report=(1, 5),
        )
    )
    console.print(
        f"Second generation created with {len(next_pop_2)} individuals."
    )
    console.print(
        f"Generation 2 - Best Fitness (Neg Loss): {best_fit_2:.4f}, Avg Fitness: {avg_fit_2:.4f}"
    )
    if best_state_dict_2:
        console.print(
            f"  Best model state dict (gen 2) keys: {list(best_state_dict_2.keys())[:3]}"
        )
    console.print(
        f"Best Offspring Accuracies: Top-1: {best_accs_2.get(1,0):.2f}%, Top-5: {best_accs_2.get(5,0):.2f}%"
    )

    console.rule(
        "[bold green]Evolutionary Components Test Finished[/bold green]"
    )
