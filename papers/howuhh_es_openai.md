# Community Implementation of OpenAI ES (Howuhh/evolution_strategies_openai)

**Link:** [https://github.com/Howuhh/evolution_strategies_openai](https://github.com/Howuhh/evolution_strategies_openai)
**Original Paper:** [Evolution Strategies as a Scalable Alternative to Reinforcement Learning (arXiv:1703.03864)](https://arxiv.org/abs/1703.03864)

## Overview
This repository provides an implementation of the OpenAI Evolution Strategies paper. It is intended for **educational purposes** and is not a distributed version as described in the original paper.

## Example Usage
```python
from training import run_experiment, render_policy

example_config = {
    "experiment_name": "test_BipedalWalker_v0",
    "plot_path": "plots/",
    "model_path": "models/", # optional
    "log_path": "logs/", # optional
    "init_model": "models/test_BipedalWalker_v5.0.pkl",  # optional
    "env": "BipedalWalker-v3",
    "n_sessions": 128,
    "env_steps": 1600, 
    "population_size": 256,
    "learning_rate": 0.06,
    "noise_std": 0.1,
    "noise_decay": 0.99, # optional
    "lr_decay": 1.0, # optional
    "decay_step": 20, # optional
    "eval_step": 10, 
    "hidden_sizes": (40, 40)
  }

policy = run_experiment(example_config, n_jobs=4, verbose=True)

# to render policy perfomance
render_policy(model_path, env_name, n_videos=10)
```

## Implemented Features
*   OpenAI ES algorithm (Algorithm 1 from the paper).
*   **Z-normalization fitness shaping** (not rank-based).
*   **Parallelization with joblib**.
*   Training examples for 6 OpenAI Gym environments (3 reportedly solved).
*   A simple three-layer neural network as an example policy.
*   Learning rate decay and noise standard deviation decay.

## Experiments and Observations
*   **CartPole:** Solved quickly. Controlling learning rate (lower is better) and noise std is important.
*   **LunarLander:** Solved. Similar to CartPole, a small learning rate is important, with slightly increased noise std.
*   **LunarLanderContinuous:** Solved much faster and better than the discrete version, possibly due to denser rewards. The agent learned to land faster and delay engine use.
*   **MountainCarContinuous:** Not solved at the time of writing. The main problem is sparse reward in the discrete version and lack of exploration in the continuous version (agent learns to stand still).
    *   Possible solution: Novelty search.
*   **BipedalWalker:** Not solved at the time of writing. More iterations needed.

## Topics
*   Reinforcement Learning
*   OpenAI Gym
*   Evolutionary Algorithms
*   Evolution Strategies
*   Implementation of Research Paper

## Languages
*   Python: 100.0% 