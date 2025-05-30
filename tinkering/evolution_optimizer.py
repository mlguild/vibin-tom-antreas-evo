import torch
import torch.nn as nn
import copy
from collections import OrderedDict
import random  # May not be needed for simple truncation but good to have

from rich.console import Console

console = Console()


class PopulationOptimizer:
    def __init__(
        self,
        population_size: int,
        model_create_fn,
        mutation_strength: float,
        device: torch.device,
    ):
        """
        Manages a population of models for a simple evolutionary algorithm
        based on mutation and truncation selection.
        """
        self.population_size = population_size
        self.model_create_fn = model_create_fn
        self.mutation_strength = mutation_strength
        self.device = device

        # self.population will be a list of nn.Module instances for simplicity of mutation and vmap prep
        # We will deepcopy them as needed.
        self.population: list[nn.Module] = self._initialize_population()

        # A CPU model instance for architectural reference in functional_call if needed by evaluators
        self.base_model_architecture_cpu = self.model_create_fn().cpu()

        console.print(
            f"[PopulationOptimizer] Initialized. Population size: {self.population_size}, Mutation strength: {self.mutation_strength}"
        )

    def _initialize_population(self) -> list[nn.Module]:
        population = []
        for _ in range(self.population_size):
            model = self.model_create_fn().to(self.device)
            # Parameters are already randomly initialized by PyTorch layers
            population.append(model)
        console.print(
            f"[PopulationOptimizer] Initialized population of {len(population)} model instances on {self.device}."
        )
        return population

    def get_current_population_individuals(self) -> list[nn.Module]:
        """Returns the current list of model instances in the population."""
        # Return copies if they are to be mutated externally, or originals if mutation happens internally
        # For our flow, run_one_generation will call apply_mutation on these.
        return self.population

    def apply_mutation_to_current_population(self) -> list[nn.Module]:
        """
        Creates a new list of offspring by mutating the current population.
        Each offspring is a deepcopy of a parent, then mutated.
        """
        offspring_population = []
        for parent_model in self.population:
            offspring_model = copy.deepcopy(parent_model).to(
                self.device
            )  # Ensure it's on the right device
            with torch.no_grad():
                for param in offspring_model.parameters():
                    if param.requires_grad:
                        noise = (
                            torch.randn_like(param, device=self.device)
                            * self.mutation_strength
                        )
                        param.add_(noise)
            offspring_population.append(offspring_model)
        return offspring_population

    def update_population(
        self,
        evaluated_offspring_population: list[nn.Module],
        fitness_scores: torch.Tensor,
    ):
        """
        Updates the internal population by selecting the top N individuals
        from the evaluated offspring population based on fitness scores.
        Assumes fitness_scores correspond to evaluated_offspring_population.
        The new population will have self.population_size individuals.
        """
        if not evaluated_offspring_population or fitness_scores.numel() == 0:
            console.print(
                "[PopulationOptimizer warning] Attempted to update population with empty offspring or fitness scores.",
                style="yellow",
            )
            return

        num_to_select = self.population_size

        # Sort by fitness scores in descending order (higher is better)
        # Argsort returns indices that would sort the original tensor
        sorted_indices = torch.argsort(fitness_scores, descending=True)

        new_population = []
        for i in range(
            min(num_to_select, len(sorted_indices))
        ):  # Ensure we don't go out of bounds
            winner_idx = sorted_indices[i].item()
            # Selected individuals are deepcopied to ensure the new generation is distinct
            new_population.append(
                copy.deepcopy(evaluated_offspring_population[winner_idx])
            )

        if (
            len(new_population) < self.population_size
            and len(new_population) > 0
        ):
            # If not enough unique best individuals, fill by duplicating the best ones found
            console.print(
                f"[PopulationOptimizer warning] Not enough distinct best offspring ({len(new_population)}) to fill population of {self.population_size}. Duplicating best.",
                style="yellow",
            )
            while len(new_population) < self.population_size:
                new_population.append(
                    copy.deepcopy(new_population[0])
                )  # Duplicate the best
        elif not new_population:
            console.print(
                "[PopulationOptimizer error] No individuals selected for new population. This should not happen if offspring were provided.",
                style="red",
            )
            # Fallback: re-initialize (though this is a sign of a deeper issue upstream)
            # self.population = self._initialize_population()
            return  # Avoid overwriting population with empty if something went very wrong

        self.population = new_population
        # console.print(f"[PopulationOptimizer] Population updated with {len(self.population)} individuals.")

    def get_best_individual_from_evaluated_offspring(
        self,
        evaluated_offspring_population: list[nn.Module],
        fitness_scores: torch.Tensor,
    ) -> tuple[nn.Module | None, float | None]:
        """
        Returns a deepcopy of the best model instance and its fitness score
        from a list of evaluated offspring.
        """
        if not evaluated_offspring_population or fitness_scores.numel() == 0:
            return None, None

        best_fitness = fitness_scores.max().item()
        best_idx = fitness_scores.argmax().item()
        best_individual_instance = copy.deepcopy(
            evaluated_offspring_population[best_idx]
        )
        return best_individual_instance, best_fitness


if __name__ == "__main__":
    from rich.traceback import install as install_rich_traceback

    install_rich_traceback()
    from tinkering.models import ResNet4StageCustom  # For testing

    console.rule("[bold green]Testing PopulationOptimizer[/bold green]")
    device_test = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pop_size = 10
    mutation_strength_test = 0.1

    def _model_creator():
        return ResNet4StageCustom(num_classes=10, in_channels=1)

    optimizer = PopulationOptimizer(
        population_size=pop_size,
        model_create_fn=_model_creator,
        mutation_strength=mutation_strength_test,
        device=device_test,
    )
    console.print(
        f"PopulationOptimizer Initialized. Pop size: {len(optimizer.population)}"
    )

    # --- Test apply_mutation_to_current_population ---
    console.rule("Test apply_mutation_to_current_population")
    offspring_models = optimizer.apply_mutation_to_current_population()
    assert len(offspring_models) == pop_size
    console.print(f"Generated {len(offspring_models)} offspring models.")
    # Check if params are different (stochastic, but highly likely with noise)
    if pop_size > 0 and len(offspring_models) > 0:
        original_param = next(
            optimizer.population[0].parameters()
        ).data.clone()
        mutated_param = next(offspring_models[0].parameters()).data
        assert not torch.allclose(
            original_param, mutated_param
        ), "Mutation did not change parameters!"
        console.print("Offspring parameters appear mutated.")

    # --- Test update_population ---
    console.rule("Test update_population")
    # Simulate fitness scores for the offspring (some high, some low)
    dummy_offspring_fitness = torch.rand(pop_size, device=device_test) * 10
    # Ensure at least one has higher fitness for selection to be meaningful if pop_size > 1
    if pop_size > 1:
        dummy_offspring_fitness[0] = 100.0
    console.print(f"Dummy offspring fitness scores: {dummy_offspring_fitness}")

    # Keep track of the state_dict of the expected best model (offspring[0] due to fitness[0]=100)
    expected_best_offspring_sd = None
    if len(offspring_models) > 0:  # Check if offspring_models is not empty
        expected_best_offspring_sd = copy.deepcopy(
            offspring_models[
                torch.argsort(dummy_offspring_fitness, descending=True)[
                    0
                ].item()
            ].state_dict()
        )

    optimizer.update_population(offspring_models, dummy_offspring_fitness)
    console.print(
        f"Population updated. New pop size: {len(optimizer.population)}"
    )

    # Verify the best model from offspring is now in the population
    if pop_size > 0 and expected_best_offspring_sd is not None:
        new_pop_sds = [m.state_dict() for m in optimizer.population]
        found_best = False
        for sd in new_pop_sds:
            match = all(
                torch.allclose(sd[key], expected_best_offspring_sd[key])
                for key in sd
            )
            if match:
                found_best = True
                break
        assert (
            found_best
        ), "The best offspring was not found in the new population after update."
        console.print(
            "Best offspring correctly selected into the new population."
        )

    # --- Test get_best_individual_from_evaluated_offspring ---
    console.rule("Test get_best_individual_from_evaluated_offspring")
    best_indiv_instance, best_fit_score = (
        optimizer.get_best_individual_from_evaluated_offspring(
            offspring_models, dummy_offspring_fitness
        )
    )
    if best_indiv_instance is not None:
        console.print(
            f"Retrieved best individual with fitness: {best_fit_score:.4f}"
        )
        assert isinstance(best_indiv_instance, nn.Module)
        assert best_fit_score == dummy_offspring_fitness.max().item()
    else:
        console.print(
            "No best individual retrieved (population or fitness was empty)."
        )

    console.rule("[bold green]PopulationOptimizer Test Finished[/bold green]")
