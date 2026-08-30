"""
Поправка на "modality collapse" кај Fusion (advanced студија, продолжение
на experiments/observation_study.py) - емпириски потврдено (30/30
метод×seed×агент комбинации, agents/networks.py::drop_branch коментар)
дека обичниот Fusion, тренирана на стандарден начин, речиси целосно ја
игнорира сликата и се потпира само на kinematics-гранката (Q-вредностите
остануваат речиси непроменети дури и кога сликата е целосно црна).

Оваа скрипта тренира "поправена" Fusion варијанта (`fusion_fixed`) - ГИ
КОМБИНИРА двете независни поправки (train.py::run_training):
  1. modality_dropout_prob=0.3 - повремено гасење на kin-гранката за
     време на тренирањето, го принудува агентот понекогаш да одлучи
     САМО од сликата.
  2. pretrained_pixels_prefix - CNN "warm-start" од веќе-истрениран
     Pixels-only модел (ИСТ seed - за да не се внесе seed-to-seed
     confound), наместо случајна CNN иницијализација.

Брз (800 епизоди, 1 seed) дијагностички тест веќе покажа дека двете
заедно даваат далеку најдобар резултат (img_sens ~0.0002->1.0+, види
разговорот/тезата) - ова е целосниот run (исти episode/seed бројки како
observation_study.py) за финални, репортабилни бројки.

Употреба:
    python experiments/fusion_fix_study.py --episodes 2000 --eval_episodes 200 --seeds 0 1 2 3 4 --workers 8

ВАЖНО за --workers: default (сите јадра - 2) МОЖЕ да ја надмине RAM на
машината - Fusion states секогаш носат слика (~100x поголема од
kinematics опсервација по запис) - види PIXEL_BUFFER_SIZE подолу.
--workers 8 е тестирано безбедно на 16-јадрена/31GB машина.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

sys.path.insert(0, str(Path(__file__).parent.parent))

from train import EVAL_SEED_OFFSET, RESULTS_DIR, run_training
from evaluate import evaluate
from experiments.parallel import agg_stats, run_jobs

METHODS = ["iql", "vdn"]
MODALITY_DROPOUT_PROB = 0.3
# Иста RAM причина како во experiments/observation_study.py::PIXEL_BUFFER_SIZE
# (Fusion states секогаш носат слика - replay buffer со default 50_000
# capacity лесно бара неколку GB по paralelen worker процес).
PIXEL_BUFFER_SIZE = 20_000


def _run_one_job(job):
    # top-level функција (не lambda/closure) - multiprocessing на Windows
    # користи "spawn", истата причина како во run_comparison.py::_run_one_job.
    method, seed, episodes, eval_episodes, num_agents = job
    pretrained_prefix = str(RESULTS_DIR / f"{method}_pixels_seed{seed}_model")  # ИСТ seed - без confound
    run_training(
        method=method, num_agents=num_agents, episodes=episodes, seed=seed,
        tag=f"fixed_seed{seed}", obs_mode="fusion",
        modality_dropout_prob=MODALITY_DROPOUT_PROB,
        pretrained_pixels_prefix=pretrained_prefix,
        agent_kwargs={"buffer_size": PIXEL_BUFFER_SIZE},
    )
    prefix = str(RESULTS_DIR / f"{method}_fusion_fixed_seed{seed}_model")
    summary = evaluate(
        method, prefix, num_agents=num_agents, episodes=eval_episodes,
        seed=EVAL_SEED_OFFSET + seed, tag=f"fixed_seed{seed}", obs_mode="fusion",
    )
    summary["seed"] = seed
    return method, summary


def _load_baseline_fusion(seeds: list[int]) -> dict[str, list[dict]]:
    """Ги вчитува веќе-готовите (НЕ-поправени) Fusion резултати за споредба, од results/observation_comparison.json."""
    path = RESULTS_DIR / "observation_comparison.json"
    if not path.exists():
        print(f"[fusion_fix_study] ПРЕДУПРЕДУВАЊЕ: {path} не постои - "
              f"нема baseline Fusion за споредба (прво пушти experiments/observation_study.py).")
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    out = {}
    for method in METHODS:
        evals = data["per_seed"].get(f"fusion/{method}", [])
        matching = [e for e in evals if e["seed"] in seeds]
        if matching:
            out[method] = matching
    return out


def main(episodes: int, eval_episodes: int, seeds: list[int], num_agents: int, max_workers: int | None):
    jobs = [(method, seed, episodes, eval_episodes, num_agents) for method in METHODS for seed in seeds]
    job_results = run_jobs(_run_one_job, jobs, max_workers=max_workers, label="метод×seed (fusion_fixed)")

    fixed_results: dict[str, list[dict]] = {m: [] for m in METHODS}
    for method, summary in job_results:
        fixed_results[method].append(summary)
    for m in fixed_results:
        fixed_results[m].sort(key=lambda s: s["seed"])

    baseline_results = _load_baseline_fusion(seeds)

    print("\n\n===== Fusion: baseline (наивен) наспроти fixed (dropout+warmstart) =====")
    print(f"{'Варијанта':<20}{'Метод':<8}{'Награда':<20}{'Стапка судири':<18}{'Стапка успех':<15}{'Просечни чекори':<18}")
    aggregate = {}
    for variant, results in [("fusion (baseline)", baseline_results), ("fusion_fixed", fixed_results)]:
        for method, evals in results.items():
            if not evals:
                continue
            row = {
                "reward": agg_stats([e["mean_reward"] for e in evals]),
                "collision_rate": agg_stats([e["collision_rate"] for e in evals]),
                "arrival_rate": agg_stats([e["arrival_rate"] for e in evals]),
                "avg_steps": agg_stats([e["avg_steps"] for e in evals]),
            }
            aggregate.setdefault(variant, {})[method] = row

            def fmt(stats):
                return f"{stats['mean']:.2f} ± {stats['ci95']:.2f}"

            print(f"{variant:<20}{method:<8}{fmt(row['reward']):<20}{fmt(row['collision_rate']):<18}"
                  f"{fmt(row['arrival_rate']):<15}{fmt(row['avg_steps']):<18}")

    out_path = RESULTS_DIR / "fusion_fix_comparison.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "modality_dropout_prob": MODALITY_DROPOUT_PROB,
                "per_seed": {"fusion_baseline": baseline_results, "fusion_fixed": fixed_results},
                "aggregate": aggregate,
            },
            f, indent=2, ensure_ascii=False,
        )
    print(f"\nЦелосните резултати зачувани во: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--eval_episodes", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--num_agents", type=int, default=3)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    main(args.episodes, args.eval_episodes, args.seeds, args.num_agents, args.workers)
