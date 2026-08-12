*[Прочитај на македонски / Read in Macedonian → README.md](README.md)*

# Multi-Agent Reinforcement Learning for Autonomous Vehicle Coordination at an Intersection

A project for the **Agent-Based Systems** course
(MDP → Bellman/DP → RL basics → Deep RL/DQN → Multi-Agent RL).

## Research question

Do agents trained with **CTDE** (Centralized Training, Decentralized
Execution - specifically **VDN, Value Decomposition Networks**) reach a
better trade-off between safety (collision rate) and efficiency (average
steps to destination / arrival rate) compared to **Independent
Q-Learning (IQL)** and a **rule-based baseline**, in a mixed-traffic
intersection scenario (controlled vehicles + human-driven vehicles)?

## My contribution

1. **A custom wrapper** (`envs/multi_agent_intersection.py`) around
   `highway-env`'s `intersection-v2`, which decomposes the standard
   shared (cooperative) reward into **individual per-agent rewards** -
   highway-env doesn't support this out of the box through its standard
   gymnasium interface.
2. **A VDN implementation from scratch** (`agents/vdn_agent.py`), built
   directly from the course theory (lecture note 10 - "VDN and QMIX
   (value decomposition for cooperative teams)").
3. **"Courtesy" (fairness) reward shaping** - an added shaping term that
   penalizes aggressive driving close to human-driven vehicles
   (`courtesy_weight` in `envs/multi_agent_intersection.py`).
4. **A systematic, statistically rigorous comparison** (multiple seeds,
   95% confidence intervals, parallel execution) across 3 approaches +
   a baseline, plus ablation, scalability, and hyperparameter studies.

## Project structure

```
traffic-marl/
├── envs/
│   └── multi_agent_intersection.py   # wrapper (key contribution)
├── agents/
│   ├── networks.py                   # Q-network, Dueling Q-network, replay buffers
│   ├── dqn_agent.py                  # DQN / Double DQN / Dueling DQN
│   ├── iql_manager.py                # Independent Q-Learning
│   ├── vdn_agent.py                  # VDN - the CTDE approach (key contribution)
│   └── heuristic_agent.py            # rule-based baseline
├── train.py                          # trains a single method
├── evaluate.py                       # evaluates a trained model
├── experiments/
│   ├── parallel.py                   # parallel execution of seeds/configs
│   ├── run_comparison.py             # main comparison (all methods × seeds)
│   ├── reward_ablation.py            # ablation: collision_reward, courtesy_weight
│   ├── scalability_study.py          # ablation: number of controlled vehicles
│   └── hyperparameter_study.py       # ablation: learning_rate/gamma/epsilon
├── report/
│   └── mdp_formalization.md          # formal MDP/Markov Game write-up
├── tests/                            # unit + integration tests
├── visualize_results.py              # generates 7 figures for the report
└── results/                          # JSON results + figures + saved models
```

## Install and quick test

```bash
pip install -r requirements.txt

python train.py --method vdn --episodes 200
python evaluate.py --method vdn --episodes 50
```

## Tests

```bash
python -m unittest discover -s tests -v
```

31 unit + integration tests, under 10 seconds total - cover the replay
buffers, epsilon decay, courtesy shaping (including MC/DC-derived cases
and boundary values), `agg_stats` (including fail-fast on empty input),
the `tag`-based result-file naming (needed for parallel execution, with
guaranteed cleanup even if a test fails mid-run), VDN target masking,
the double-DQN branch (with a fault-revealing comparison, not just
"doesn't crash"), `HeuristicAgent` branch/boundary coverage, that
rewards are genuinely different per agent (not just same shape/type),
the leaked-arrival-reward fix, and `MultiAgentIntersectionEnv` against
the real highway-env.

## Evaluation metrics

- **mean_reward** - average reward per agent (± 95% CI across seeds)
- **collision_rate** - fraction of episodes with at least one collision (safety)
- **arrival_rate** - fraction of agents that successfully reached their destination (efficiency)
- **avg_steps** - average episode length (throughput)

## Experiments

Every (method, seed) / (config, seed) is an independent job, run in
PARALLEL via `experiments/parallel.py` (multiple CPU cores instead of
one at a time - `--workers N` to control it manually, `--workers 1` for
sequential).

```bash
# main comparison: heuristic / IQL / VDN, multiple seeds
python experiments/run_comparison.py --episodes 1500 --eval_episodes 150 --seeds 0 1 2 3 4

# ablation: collision_reward weight + courtesy_weight
python experiments/reward_ablation.py --episodes 800 --eval_episodes 80 --seeds 0 1 2

# scalability: 2-5 controlled vehicles
python experiments/scalability_study.py --agent_counts 2 3 4 5 --episodes 800 --eval_episodes 80 --seeds 0 1 2

# hyperparameter study (VDN): learning_rate / discount_factor / epsilon
python experiments/hyperparameter_study.py --episodes 800 --eval_episodes 80 --seeds 0 1 2
```

## Visualization

```bash
python visualize_results.py
```

Generates 7 figures in `results/figures/` (300 DPI, consistent color
palette per method): `training_curves.png`, `method_comparison.png`,
`radar_comparison.png`, `reward_ablation.png`, `courtesy_ablation.png`,
`scalability.png`, `hyperparameter_study.png`.
