# Evolution Strategies as a Scalable Alternative to Reinforcement Learning

**Authors:** Tim Salimans, Jonathan Ho, Xi Chen, Szymon Sidor, Ilya Sutskever

**arXiv Link:** [https://arxiv.org/abs/1703.03864](https://arxiv.org/abs/1703.03864)

**Date:** Submitted on 10 Mar 2017 (v1), last revised 7 Sep 2017 (v2)

## Abstract

We explore the use of Evolution Strategies (ES), a class of black box optimization algorithms, as an alternative to popular MDP-based RL techniques such as Q-learning and Policy Gradients. Experiments on MuJoCo and Atari show that ES is a viable solution strategy that scales extremely well with the number of CPUs available: By using a novel communication strategy based on common random numbers, our ES implementation only needs to communicate scalars, making it possible to scale to over a thousand parallel workers. This allows us to solve 3D humanoid walking in 10 minutes and obtain competitive results on most Atari games after one hour of training. In addition, we highlight several advantages of ES as a black box optimization technique: it is invariant to action frequency and delayed rewards, tolerant of extremely long horizons, and does not need temporal discounting or value function approximation.

## Key Takeaways (from Abstract)

*   ES is presented as an alternative to Q-learning and Policy Gradients for RL.
*   Scales very well with the number of CPUs (tested with over a thousand parallel workers).
*   Achieved fast results: 3D humanoid walking solved in 10 minutes; competitive Atari results in one hour.
*   Communication strategy relies on common random numbers, only requiring scalar communication (low bandwidth).
*   Advantages of ES as a black box method:
    *   Invariant to action frequency and delayed rewards.
    *   Tolerant of extremely long horizons.
    *   Does not need temporal discounting.
    *   Does not need value function approximation.

## Further Notes

*(Full paper content could not be automatically ingested. This summary is based on the abstract.)* 