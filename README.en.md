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
5. **An advanced additional study - 3 observation modes** (Kinematics /
   Pixels / Fusion, see section below), tested with both IQL and VDN,
   plus a **robustness study** (noise/blur/darken/branch-dropout vs.
   clean evaluation). The project's main topic remains VDN/CTDE
   coordination with Kinematics observations (points 1-4 above) - this
   is an extension, not a replacement.

## Project structure

```
traffic-marl/
├── envs/
│   └── multi_agent_intersection.py   # wrapper (key contribution) - Kinematics/Pixels/Fusion obs_mode
├── agents/
│   ├── networks.py                   # Q-network, Dueling/Pixel/Fusion Q-networks, replay buffers
│   ├── dqn_agent.py                  # DQN / Double DQN / Dueling DQN
│   ├── iql_manager.py                # Independent Q-Learning
│   ├── vdn_agent.py                  # VDN - the CTDE approach (key contribution)
│   └── heuristic_agent.py            # rule-based baseline
├── train.py                          # trains a single method (--obs_mode kinematics/pixels/fusion)
├── evaluate.py                       # evaluates a trained model (+ optional robustness perturb_fn)
├── experiments/
│   ├── parallel.py                   # parallel execution of seeds/configs
│   ├── run_comparison.py             # main comparison (all methods × seeds, Kinematics)
│   ├── reward_ablation.py            # ablation: collision_reward, courtesy_weight
│   ├── scalability_study.py          # ablation: number of controlled vehicles
│   ├── hyperparameter_study.py       # ablation: learning_rate/gamma/epsilon
│   ├── significance_test.py          # Welch's t-test over comparison_summary.json
│   ├── observation_study.py          # advanced: Pixels/Fusion × IQL/VDN training + comparison
│   ├── robustness.py                 # advanced: pure perturbation functions (noise/blur/darken/dropout)
│   ├── robustness_study.py           # advanced: robustness evaluation of the already-trained models
│   ├── fusion_fix_study.py           # advanced: "modality collapse" finding + dropout/warm-start fix
│   └── modality_sensitivity.py       # advanced: img_sens/kin_sens Q-value diagnostic (Fusion)
├── report/
│   └── mdp_formalization.md          # formal MDP/Markov Game write-up
├── tests/                            # unit + integration tests
├── visualize_results.py              # generates 9 figures for the report
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

66 unit + integration tests (`tests/test_core.py` - 34, the original
Kinematics/heuristic/IQL/VDN pipeline; `tests/test_observations.py` -
32, the Pixels/Fusion extension), under 30 seconds total - cover the
replay buffers, epsilon decay, courtesy shaping (including MC/DC-derived
cases and boundary values), `agg_stats` (including fail-fast on empty
input), the `tag`-based result-file naming (needed for parallel
execution, with guaranteed cleanup even if a test fails mid-run), VDN
target masking, the double-DQN branch (with a fault-revealing
comparison, not just "doesn't crash"), `HeuristicAgent` branch/boundary
coverage, that rewards are genuinely different per agent (not just same
shape/type), the leaked-arrival-reward fix, that the SAME --seed produces
an IDENTICAL training history (regression test for a bug found during
the final review: `random.sample()` in the replay buffers wasn't seeded
- see `train.py::set_global_seed`), `MultiAgentIntersectionEnv` against
the real highway-env, **that two different agents' pixel observations in
the SAME step are NOT identical** (directly enforces the "own local
image per agent" requirement), the Pixel/Fusion Q-networks,
`MultiAgentReplayBuffer` with fusion states, IQL/VDN train_step() for
pixels/fusion, the robustness perturbation functions
(noise/blur/darken/branch-dropout), the modality-dropout/CNN-warm-start
fix for Fusion, and that `env.close()` is ALWAYS called - even when
`run_training()`/`evaluate()` raise partway through.

## Evaluation metrics

- **mean_reward** - average reward per agent (± 95% CI across seeds)
- **collision_rate** - fraction of episodes with at least one collision (safety)
- **arrival_rate** - fraction of agents that successfully reached their destination (efficiency)
- **avg_steps** - average episode length (throughput)

## Observation modes (Kinematics / Pixels / Fusion) - advanced study

The project's main topic (the research question above) is VDN/CTDE
coordination with **Kinematics** observations. As an advanced additional
study, the same IQL/VDN pipeline (same env settings, seeds,
training/evaluation pipeline, metrics) is extended with 2 more
observation types:

1. **Kinematics-only** (default, unchanged) - a flattened vector of
   [presence, x, y, vx, vy, cos_h, sin_h] for the 6 nearest vehicles.
2. **Pixels-only** - each agent gets its **own, local, agent-centric
   grayscale image** (a stack of 4 frames, so the CNN can "see" velocity
   via frame-to-frame differences), processed through a CNN. **This is a
   deliberate design decision**: highway-env's `MultiAgentObservation`
   creates a SEPARATE `GrayscaleObservation` instance per controlled
   vehicle and centers the camera on exactly that vehicle
   (`observer_vehicle`) - each agent genuinely sees its own image, NOT
   the same global screenshot of the whole intersection shared by all
   agents (verified by
   `tests/test_observations.py::TestEnvPixelObservations`).
3. **Fusion** - a local kinematics vector + a local grayscale image, an
   MLP branch + a CNN branch merged before the final Q-values.

The architectures (`agents/networks.py`) and the IQL/VDN algorithmic
logic (double-DQN masking, VDN target decomposition) are EXTENDED, not
replaced - the same existing networks/replay buffers, just with added
`obs_mode`/`img_shape` parameters (no Stable-Baselines3 used).

```bash
# train/evaluate a single method with a specific observation mode
python train.py --method vdn --episodes 200 --obs_mode pixels
python evaluate.py --method vdn --episodes 50 --obs_mode pixels

# full advanced study: Pixels+Fusion × IQL/VDN (Kinematics×IQL/VDN is
# reused from run_comparison.py, no retraining) + a unified table
#
# --workers 8 (not the default, which would take all cores-2): pixels/
# fusion replay buffers hold images (~100x bigger than kinematics) - too
# many parallel processes easily exceed available RAM (tested: 8 workers
# is safe on a 31GB machine - adjust to your own RAM).
python experiments/observation_study.py --episodes 2000 --eval_episodes 200 --seeds 0 1 2 3 4 --workers 8
```

### Robustness study

The same already-trained models (Kinematics/Pixels/Fusion × IQL/VDN) are
evaluated under 3 types of observation perturbation (ALWAYS only at
evaluation time - the models are trained exclusively on clean
observations):

- **kin_noise** - Gaussian noise on the kinematics vector (simulates
  imprecise GPS/radar readings)
- **pixel_blur** / **pixel_dark** - Gaussian blur / darkening of the
  pixel image (fog/rain, dusk)
- **drop_kin** / **drop_img** - temporarily disabling one observation
  branch (Fusion only - simulates a sensor outage)

Metrics measured: mean_reward, collision_rate, arrival_rate, avg_steps,
and **% drop vs. the clean evaluation** (paired by seed - the same
model, the same seeds, with and without the perturbation).

```bash
python experiments/robustness_study.py --seeds 0 1 2 3 4 --eval_episodes 200
```

### "Modality collapse" finding and an attempted fix (`experiments/fusion_fix_study.py`)

The robustness study turned up an odd result: `pixel_blur`/`pixel_dark`
for Fusion came out at **~0% drop**, even though the image was
genuinely being changed. A direct inspection of the Q-values
(`experiments/modality_sensitivity.py` - 30/30 method×seed×agent
combinations, `agents/networks.py::drop_branch`) confirmed why: the
Fusion network, trained the ordinary way, **almost completely ignores
the image** and relies only on kinematics (Q-values stay nearly
unchanged even when the image is fully black - `img_sens` ≈ 0.00-0.04,
versus `kin_sens` ≈ 2.3-4.0 when the kinematics branch is zeroed
instead - i.e. mean ratio kin/img ≈ 1100-2100×). This is a known
multimodal-learning phenomenon ("modality collapse"/"gradient
starvation") - when one branch (kinematics: clean, low-dimensional,
quickly useful) is much easier to learn from than the other (pixels:
needs far more samples before the CNN extracts anything useful), the
network quickly stops learning through the harder branch.

`experiments/fusion_fix_study.py` trains a "fixed" Fusion variant
(`obs_mode="fusion"`, `modality_dropout_prob=0.3` + a CNN
`pretrained_pixels_prefix` warm-start from a Pixels-only model, see
`train.py::run_training`) and compares it against the original (all
numbers below are from a full 5-seed retrain, reproducible via the
commands below):

| | img_sens (0=ignores image) | mean_reward | arrival_rate (clean) |
|---|---|---|---|
| Fusion (plain) | ≈0.00-0.04 | 5.9-6.0 | 0.44-0.48 |
| Fusion (fixed) | 0.6-8.6 (per-method mean; individual combos 0-105, high agent/seed variance) | 4.8-5.5 | 0.21-0.36 |

Under `kin_noise`/`drop_kin`, plain Fusion loses 5-24% arrival_rate
(paired by seed - see `results/robustness_study.json`). For the fixed
Fusion, arrival_rate under the same two conditions **does not drop** -
on average it actually RISES relative to its own (lower) clean baseline
(IQL: +66% under `kin_noise`, +212% under `drop_kin`; VDN: +2% / +24%;
high seed-to-seed variance, especially for IQL).

Conclusion: the fix genuinely forces the network to use the image for
most (not all) agent×seed combinations - img_sens rises by ~2-3 orders
of magnitude on average, but with noticeable instability (some agents
still stay close to collapse even with the fix). This comes at a cost -
lower clean-condition performance (reward and arrival_rate both drop),
in exchange for potentially more resilient behavior when the kinematics
sensor is degraded/unavailable. This is a classic robustness-vs-
performance trade-off, not a free improvement - so the "fixed" Fusion
does NOT replace plain Fusion in the main comparison above, it stands as
a separate, additional finding.

```bash
python experiments/fusion_fix_study.py --episodes 2000 --eval_episodes 200 --seeds 0 1 2 3 4 --workers 8

# reproduce the img_sens/kin_sens table above
python experiments/modality_sensitivity.py --obs_modes fusion fusion_fixed --seeds 0 1 2 3 4
```

## Experiments

Every (method, seed) / (config, seed) is an independent job, run in
PARALLEL via `experiments/parallel.py` (multiple CPU cores instead of
one at a time - `--workers N` to control it manually, `--workers 1` for
sequential).

```bash
# main comparison: heuristic / IQL / VDN, multiple seeds
python experiments/run_comparison.py --episodes 1500 --eval_episodes 150 --seeds 0 1 2 3 4

# significance test (Welch's t-test) over comparison_summary.json
python experiments/significance_test.py

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

Generates 9 figures in `results/figures/` (300 DPI, consistent color
palette per method): `training_curves.png`, `method_comparison.png`,
`radar_comparison.png`, `reward_ablation.png`, `courtesy_ablation.png`,
`scalability.png`, `hyperparameter_study.png`, `obs_mode_comparison.png`
(Kinematics/Pixels/Fusion × IQL/VDN), `robustness_study.png` (% drop
under each perturbation). Each advanced-study figure is skipped
gracefully if the corresponding JSON doesn't exist yet
(`observation_study.py`/`robustness_study.py` haven't been run yet).
