import torch
import torch.nn as nn
from torch.func import functional_call, vmap, stack_module_state
import copy
import random
import time

from rich.console import Console
from rich.pretty import pprint

console = Console()

# --- Fitness & Accuracy Metrics ---
_criterion_for_loss_metric = (
    nn.CrossEntropyLoss()
)  # Used if fitness is loss-based


def negative_loss_fitness_metric(
    outputs: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Calculates negative cross-entropy loss. Higher is better."""
    loss = _criterion_for_loss_metric(outputs.float(), labels)
    return -loss


def accuracy_metric(
    outputs: torch.Tensor, labels: torch.Tensor, top_k: tuple = (1,)
) -> dict[int, torch.Tensor]:
    """Calculates top-k accuracy. Returns a dict {k: accuracy_tensor}."""
    max_k = max(top_k)
    batch_size = labels.size(0)
    if batch_size == 0:  # Avoid division by zero for empty batches
        return {k_val: torch.tensor(0.0) for k_val in top_k}

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
    # console.print(f"[info]Initialized population with {population_size} individuals on {device}.")
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
        eval_start_time = time.time()
        if not population_models_list:
            return torch.empty(0, device=device)

        inputs, labels = data_batch_tuple
        console.print(
            f"[eval_pop_batch DBG] Input shape: {inputs.shape}, Labels shape: {labels.shape}, Device: {inputs.device}",
            highlight=False,
        )

        t0 = time.time()
        # Ensure models are on the correct device before stacking state
        # This should ideally be handled when population_models_list is created/updated
        # Forcing it here as an extra check:
        # population_models_list = [m.to(device) for m in population_models_list]
        stacked_params, stacked_buffers = stack_module_state(
            population_models_list
        )
        params_and_buffers_for_vmap = (stacked_params, stacked_buffers)
        console.print(
            f"[eval_pop_batch DBG] stack_module_state took: {time.time() - t0:.4f}s",
            highlight=False,
        )

        use_autocast = amp_dtype is not None and device.type == "cuda"

        batched_outputs = None  # Initialize to avoid reference before assignment if loop is empty
        with torch.no_grad():
            with torch.amp.autocast(
                device_type=device.type,
                enabled=use_autocast,
                dtype=amp_dtype if use_autocast else torch.float32,
            ):
                t1 = time.time()
                console.print(
                    f"[eval_pop_batch DBG] Calling vmapped_get_outputs. Input device: {inputs.device}",
                    highlight=False,
                )
                if params_and_buffers_for_vmap[0]:
                    first_param_name = next(
                        iter(params_and_buffers_for_vmap[0]), None
                    )
                    if (
                        first_param_name
                        and first_param_name in params_and_buffers_for_vmap[0]
                    ):
                        console.print(
                            f"[eval_pop_batch DBG] Param '{first_param_name}' device: {params_and_buffers_for_vmap[0][first_param_name].device}",
                            highlight=False,
                        )

                batched_outputs = vmapped_get_outputs(
                    params_and_buffers_for_vmap, inputs
                )
                console.print(
                    f"[eval_pop_batch DBG] vmapped_get_outputs took: {time.time() - t1:.4f}s. Output shape: {batched_outputs.shape if batched_outputs is not None else 'None'}",
                    highlight=False,
                )

        t2 = time.time()
        fitness_scores = []
        if batched_outputs is not None:
            for i in range(len(population_models_list)):
                outputs_for_model_i = batched_outputs[i]
                fitness = fitness_metric_fn(outputs_for_model_i, labels)
                fitness_scores.append(
                    fitness.item() if hasattr(fitness, "item") else fitness
                )
        # console.print(f"[eval_pop_batch DBG] Fitness calculation loop took: {time.time() - t2:.4f}s")
        # console.print(f"[eval_pop_batch DBG] Total evaluate_population_on_batch took: {time.time() - eval_start_time:.4f}s")
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
    gen_num: int = 0,  # For logging context
):
    """
    Runs one generation: mutate, evaluate fitness (neg loss), select.
    Additionally, evaluates accuracy of the best offspring using functional_call.
    """
    gen_start_time = time.time()
    console.print(f"[Gen {gen_num} DBG] Starting generation.", highlight=False)

    t0 = time.time()
    offspring_population = apply_mutation(
        current_population_models, mutation_strength, device
    )
    console.print(
        f"[Gen {gen_num} DBG] apply_mutation took: {time.time() - t0:.4f}s, Num offspring: {len(offspring_population)}",
        highlight=False,
    )

    t1 = time.time()
    offspring_fitness_scores = fitness_eval_fn(
        offspring_population, data_batch_tuple, device
    )
    console.print(
        f"[Gen {gen_num} DBG] fitness_eval_fn (evaluate_population_on_batch) took: {time.time() - t1:.4f}s",
        highlight=False,
    )

    best_fitness_this_gen = -float("inf")
    avg_fitness_this_gen = -float("inf")
    best_offspring_state_dict = None
    best_offspring_top_k_accuracies = {k: 0.0 for k in top_k_acc_report}
    t_acc_eval_start = time.time()

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
        with (
            torch.no_grad(),
            torch.amp.autocast(
                device_type=device.type,
                enabled=use_autocast_acc,
                dtype=amp_dtype if use_autocast_acc else torch.float32,
            ),
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
    console.print(
        f"[Gen {gen_num} DBG] Accuracy eval for best offspring took: {time.time() - t_acc_eval_start:.4f}s",
        highlight=False,
    )

    t_selection_start = time.time()
    num_to_select_for_next_gen = len(current_population_models)
    next_generation_population = select_top_n_offspring(
        offspring_population,
        offspring_fitness_scores,
        num_to_select_for_next_gen,
    )
    console.print(
        f"[Gen {gen_num} DBG] select_top_n_offspring took: {time.time() - t_selection_start:.4f}s",
        highlight=False,
    )
    console.print(
        f"[Gen {gen_num} DBG] Total run_one_generation took: {time.time() - gen_start_time:.4f}s",
        highlight=False,
    )

    return (
        next_generation_population,
        best_offspring_state_dict,
        best_fitness_this_gen,
        avg_fitness_this_gen,
        best_offspring_top_k_accuracies,
    )


# --- Population Based Optimizer ---
class PopulationOptimizer:
    def __init__(
        self,
        population_size: int,
        model_create_fn,  # Callable that returns a new model instance
        mutation_strength: float,
        device: torch.device,
    ):
        self.population_size = population_size
        self.model_create_fn = model_create_fn
        self.mutation_strength = mutation_strength
        self.device = device
        self.population: list[nn.Module] = self._initialize_population()
        # For get_fitness_evaluation_fn, which needs a base model for vmap architecture
        self.base_model_architecture_template = self.model_create_fn().to(
            self.device
        )

        console.print(
            f"[PopulationOptimizer] Initialized. Pop_size: {self.population_size}, MutSigma: {self.mutation_strength}"
        )

    def _initialize_population(self) -> list[nn.Module]:
        pop = []
        for _ in range(self.population_size):
            pop.append(self.model_create_fn().to(self.device))
        console.print(
            f"[PopulationOptimizer] Initialized population of {len(pop)} model instances."
        )
        return pop

    def generate_offspring_for_evaluation(self) -> list[nn.Module]:
        """Creates a new list of offspring by mutating the current population."""
        offspring_population = []
        t_start_mutation = time.time()
        for parent_model in self.population:
            offspring_model = copy.deepcopy(
                parent_model
            )  # Stays on same device
            with torch.no_grad():
                for param in offspring_model.parameters():
                    if param.requires_grad:
                        noise = (
                            torch.randn_like(param) * self.mutation_strength
                        )
                        param.add_(noise)
            offspring_population.append(offspring_model)
        # console.print(f"[DBG] Mutation for {len(offspring_population)} took: {time.time()-t_start_mutation:.4f}s", highlight=False)
        return offspring_population

    def update_population_with_selected_offspring(
        self,
        evaluated_offspring_population: list[nn.Module],
        fitness_scores: torch.Tensor,
    ):
        """Updates the internal population by selecting the top N individuals from offspring."""
        t_start_selection = time.time()
        if not evaluated_offspring_population or fitness_scores.numel() == 0:
            console.print(
                "[PopulationOptimizer warning] Update attempt with empty offspring/fitness.",
                style="yellow",
            )
            return

        num_to_select = self.population_size
        sorted_indices = torch.argsort(fitness_scores, descending=True)

        new_population = []
        for i in range(min(num_to_select, len(sorted_indices))):
            winner_idx = sorted_indices[i].item()
            new_population.append(
                copy.deepcopy(evaluated_offspring_population[winner_idx])
            )

        if (
            len(new_population) < self.population_size
            and len(new_population) > 0
        ):
            console.print(
                f"[PopulationOptimizer warning] Filling remaining population by duplicating best ({self.population_size - len(new_population)} times).",
                style="yellow",
            )
            while len(new_population) < self.population_size:
                new_population.append(copy.deepcopy(new_population[0]))
        elif not new_population:
            console.print(
                "[PopulationOptimizer error] No individuals selected.",
                style="red",
            )
            return
        self.population = new_population
        # console.print(f"[DBG] Selection & update took: {time.time()-t_start_selection:.4f}s", highlight=False)


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
    # criterion_test = nn.CrossEntropyLoss() # Already global in the module _criterion_for_loss_metric
    # def _negative_loss_metric(outputs, labels): # Already global in the module
    #     loss = _criterion_for_loss_metric(outputs, labels)
    #     return -loss  # Negative loss, so higher is better

    console.print(
        f"Test params: pop_size={pop_size_test}, mutation_strength={mutation_strength_test}, fitness=neg_loss, device={device_test}",
        highlight=False,
    )

    # 1. Initialize Population
    console.rule("1. Initialize Population")
    current_pop = initialize_population(pop_size_test, _model_fn, device_test)
    console.print(f"Initialized population of {len(current_pop)} models.")

    # 2. Get Fitness Evaluation Function
    console.rule("2. Prepare Fitness Evaluation")
    base_model_arch_instance = _model_fn().to(device_test)
    neg_loss_fitness_evaluator = get_fitness_evaluation_fn(
        base_model_arch_instance,
        negative_loss_fitness_metric,
        amp_dtype=None,  # Test with fp32 for fitness eval
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
    s_time = time.time()
    next_pop, best_sd, best_fit, avg_fit, best_accs = run_one_generation(
        current_pop,
        base_model_arch_instance,
        mutation_strength_test,
        dummy_batch,
        neg_loss_fitness_evaluator,
        accuracy_metric,
        device_test,
        amp_dtype=None,
        top_k_acc_report=(1, 5),
        gen_num=1,  # For debug logs inside
    )
    console.print(
        f"run_one_generation (1) took: {time.time() - s_time:.4f}s",
        highlight=False,
    )
    console.print(
        f"Next gen: {len(next_pop)} ind. BestFit(NegLoss): {best_fit:.4f}, AvgFit: {avg_fit:.4f}"
    )
    console.print(
        f"Best Offspring Accs: Top-1: {best_accs.get(1,0):.2f}%, Top-5: {best_accs.get(5,0):.2f}%"
    )
    if best_sd:
        console.print(
            f"  Best model SD keys: {list(best_sd.keys())[:3]}",
            highlight=False,
        )

    console.rule("4. Run Second Generation (sanity check)")
    current_pop = next_pop
    s_time = time.time()
    next_pop_2, _, best_fit_2, avg_fit_2, best_accs_2 = run_one_generation(
        current_pop,
        base_model_arch_instance,
        mutation_strength_test,
        dummy_batch,
        neg_loss_fitness_evaluator,
        accuracy_metric,
        device_test,
        amp_dtype=None,
        top_k_acc_report=(1, 5),
        gen_num=2,  # For debug logs inside
    )
    console.print(
        f"run_one_generation (2) took: {time.time() - s_time:.4f}s",
        highlight=False,
    )
    console.print(
        f"Gen 2 - BestFit(NegLoss): {best_fit_2:.4f}, AvgFit: {avg_fit_2:.4f}"
    )
    console.print(
        f"Gen 2 Best Accs: Top-1: {best_accs_2.get(1,0):.2f}%, Top-5: {best_accs_2.get(5,0):.2f}%"
    )

    console.rule(
        "[bold green]Evolutionary Components Test Finished[/bold green]"
    )
