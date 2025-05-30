# Evolutionary Strategies (ES) - Path Forward

This document outlines the planned approach for implementing and exploring Evolutionary Strategies within this project, building upon the existing infrastructure.

## 1. Core Concepts Review

Before implementation, a quick review of key ES concepts (based on `papers/salimans_et_al_2017_es.md` and related resources):
*   **Black-box Optimization**: ES optimizes a policy (neural network parameters $\theta$) by evaluating its performance (fitness $F(\theta)$) without requiring gradients of $F$ with respect to $\theta$.
*   **Population-based**: Maintains a population of candidate parameter vectors.
*   **Perturbation (Mutation)**: At each generation, a population of parameter vectors $\theta$ is perturbed, typically by adding Gaussian noise: $\theta' = \theta + \sigma \epsilon$, where $\epsilon \sim N(0, I)$.
*   **Evaluation (Fitness Calculation)**: Each perturbed parameter vector $\theta'$ is used to run one or more episodes in the environment, and its fitness (e.g., total reward) is recorded.
*   **Update Rule**: The original parameter vector $\theta$ is updated based on the fitness scores of the perturbations. A common update is a weighted sum of the perturbations, scaled by their fitness: 
    $\theta_{t+1} \leftarrow \theta_t + \alpha \frac{1}{N \sigma} \sum_{i=1}^{N} F_i \epsilon_i$, where $N$ is the population size (number of perturbations), $\alpha$ is the learning rate.
*   **Advantages Highlighted**: Scalability, robustness to reward sparsity/delay, no need for value functions or backpropagation (computationally cheaper per parameter evaluation).

## 2. Implementation Plan

### 2.1. ES Optimizer/Algorithm Class

Create a new Python module (e.g., `tinkering/evolution.py`) to house the ES algorithm.

*   **`ESOptimizer` Class**:
    *   **Initialization (`__init__`)**: 
        *   `base_model (nn.Module)`: A template model instance to define the architecture and get initial parameters.
        *   `population_size (int)`: Number of perturbations to sample per generation (must be even if using antithetic sampling).
        *   `sigma (float)`: Standard deviation of the Gaussian noise for perturbations.
        *   `learning_rate (alpha, float)`: Learning rate for updating the central parameters.
        *   `device (torch.device)`: Device for computations.
        *   `reward_shaping_fn (callable, optional)`: Function to apply fitness shaping (e.g., ranking, z-score normalization).
    *   **`ask()` method**: 
        *   Generates `population_size` noise vectors $\epsilon_i$.
        *   Creates `population_size` perturbed parameter sets: $\theta_t + \sigma \epsilon_i$.
        *   Returns these perturbed parameter sets (e.g., as a list of state_dicts or a batched parameter structure compatible with `functional_call`).
        *   Optionally implement antithetic sampling (evaluating $\epsilon_i$ and $-\epsilon_i$).
    *   **`tell(fitness_scores)` method**: 
        *   Takes a list/tensor of fitness scores corresponding to the perturbed parameters from `ask()`.
        *   Applies reward shaping if configured.
        *   Computes the gradient estimate $\sum F_i \epsilon_i$.
        *   Updates the central parameters $\theta_t$ using the ES update rule.
    *   **`current_parameters()` method**: Returns the current central parameters $\theta_t$.

### 2.2. Evaluation Function (Fitness Calculation)

*   This function will take a set of model parameters (e.g., a state_dict), load them into our `SimpleResNet` (or any compatible model), and evaluate it on a task. For EMNIST, this would involve:
    *   Running the model on a batch (or several batches) of EMNIST validation data.
    *   The fitness could be the negative loss (to maximize) or accuracy.
*   This evaluation needs to be efficient. We will leverage the `functional_call` and `vmap` utilities from `tinkering/models.py` to evaluate a whole population of perturbed models in parallel on data batches.

### 2.3. Main ES Training Script (`run_es_based.py`)

*   Similar structure to `run_gradient_based.py` but adapted for ES.
*   Uses `fire` for CLI arguments (population_size, sigma, learning_rate, total_generations, etc.).
*   Initializes the EMNIST dataloaders (primarily for validation/fitness evaluation).
*   Initializes the `ESOptimizer` with a base `SimpleResNet` model.
*   **Main Loop (Generations)**:
    1.  Call `es_optimizer.ask()` to get a population of perturbed parameter sets.
    2.  For each parameter set in the population:
        a.  Evaluate its fitness using the EMNIST validation set (or a subset). This is where `functional_call`/`vmap` for batched model evaluation will be crucial.
           Each member of the population (a set of weights) will be evaluated on the *same* batch(es) of validation data to ensure fair comparison of fitness.
    3.  Collect all fitness scores.
    4.  Call `es_optimizer.tell(fitness_scores)` to update the central parameters.
    5.  Log progress (generation number, best fitness, average fitness, etc.).
    6.  Periodically save the best performing central parameters.

## 3. Key Considerations & Enhancements

*   **Reward Shaping**: Implement rank-based fitness shaping or z-score normalization to stabilize training.
*   **Antithetic Sampling**: Evaluate perturbations in pairs $(\epsilon, -\epsilon)$ to reduce variance.
*   **Parallelism**: 
    *   `vmap` will provide parallelism for evaluating model perturbations on the GPU.
    *   If evaluations are very expensive, consider Python's `multiprocessing` for distributing the evaluation of different population members across CPU cores (if GPU is a bottleneck or multiple GPUs aren't easily used by `vmap` for this task structure).
*   **Adaptive Sigma/Learning Rate**: Explore strategies for adapting $\sigma$ or $\alpha$ during training, although the original paper found fixed hyperparameters worked well.
*   **Comparison with Gradient-Based**: Once working, compare performance (sample efficiency, wall-clock time, final accuracy) against the results from `run_gradient_based.py`.

## 4. Path Forward - Step-by-Step

1.  ✅ **Initial Setup**: Create `docs` folder and basic documentation structure (this step).
2.  ➡️ **Implement `ESOptimizer`**: Develop the core `ESOptimizer` class in `tinkering/evolution.py`.
3.  **Develop Fitness Evaluation**: Write the function to calculate fitness for a given set of model parameters using EMNIST data and `functional_call`/`vmap`.
4.  **Create `run_es_based.py`**: Implement the main ES training loop.
5.  **Testing and Refinement**: Thoroughly test the ES implementation, debug, and refine.
6.  **Add Enhancements**: Incorporate antithetic sampling and reward shaping.
7.  **Benchmark and Compare**: Run experiments and compare with the gradient-based approach. 