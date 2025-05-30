# OpenAI Evolution Strategies Starter (openai/evolution-strategies-starter)

**Link:** [https://github.com/openai/evolution-strategies-starter](https://github.com/openai/evolution-strategies-starter)
**Paper:** [Evolution Strategies as a Scalable Alternative to Reinforcement Learning (arXiv:1703.03864)](https://arxiv.org/abs/1703.03864)

## Overview
This repository provides code for the paper "Evolution Strategies as a Scalable Alternative to Reinforcement Learning" by Tim Salimans, Jonathan Ho, Xi Chen, and Ilya Sutskever.

The implementation uses a **master-worker architecture**:
*   At each iteration, the master broadcasts parameters to the workers.
*   Workers send returns back to the master.
This architecture was used for the humanoid scaling experiment mentioned in the paper.

## Key Features
*   **Distributed Implementation:** Designed to run on EC2 (AWS).
*   **Resilient to Worker Termination:** Safe to run workers on spot instances.

## Setup and Usage

### Build AMI (Amazon Machine Image)
1.  The humanoid experiment depends on Mujoco. You need to provide your own Mujoco license and binary in `scripts/dependency.sh`.
2.  Install Packer.
3.  Build images by running `cd scripts && packer build packer.json`.
    *   You can optionally configure `scripts/packer.json` to choose build instance or AWS regions.
4.  Packer will return a list of AMI IDs. Place these in `AMI_MAP` in `scripts/launch.py`.

### Launching Experiments
*   Use `scripts/launch.py` along with an experiment JSON file.
*   An example JSON file is provided in the `configurations` directory.
*   All command-line arguments to `scripts/launch.py` must be filled in.

## Repository Status
*   **Archive:** The code is provided as-is, and no updates are expected.

## Contributors
*   Christopher Hesse (@christopherhesse)
*   Jonathan Ho (@hojonathanho)

## License
*   MIT License

## Languages
*   Python: 94.3%
*   Shell: 5.7% 