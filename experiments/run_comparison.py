"""
Оваа скрипта ја пушта целата споредбена студија потребна за проектот:
  - тренира 3 методи (heuristic нема тренинг, само евалуација) x N seed-ови
  - евалуира секој истрениран модел
  - прави агрегирана табела со средина + 95% доверителен интервал низ seed-овите

Секој (метод, seed) е целосно независен job, па сите ги пуштам ПАРАЛЕЛНО
преку experiments/parallel.py наместо еден по еден - на машина со повеќе
јадра ова е директно пропорционално побрзо (види parallel.py за детали
зошто ова е коректно и безбедно, не само "побрзо на око").

Употреба (брз "dev" режим - неколку минути):
    python experiments/run_comparison.py --episodes 150 --eval_episodes 30 --seeds 0 1 2

Употреба (целосен "проектен" режим - оставам подолго / преку ноќ):
    python experiments/run_comparison.py --episodes 2000 --eval_episodes 200 --seeds 0 1 2 3 4

Употреба (--workers 1 за секвенцијално, полезно за debug):
    python experiments/run_comparison.py --episodes 100 --seeds 0 --workers 1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Кирилски print() на не-UTF8 конзола фрла UnicodeEncodeError - истата
# причина како во train.py, види коментар таму. Ова е родителскиот процес
# (worker-ите се фиксираат одделно во experiments/parallel.py::_init_worker).
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

sys.path.insert(0, str(Path(__file__).parent.parent))

from train import run_training, RESULTS_DIR, EVAL_SEED_OFFSET
from evaluate import evaluate
from experiments.parallel import run_jobs, agg_stats

METHODS = ["heuristic", "iql", "vdn"]


def _run_one_job(job):
    # ВАЖНО: оваа функција мора да е дефинирана на "top level" од модулот
    # (не внатре во main()) - multiprocessing на Windows користи "spawn",
    # кое секој worker процес го стартува со НОВ Python интерпретер и мора
    # да може да ја "увезе" (import) оваа функција по име, не само да ја
    # добие preко затворена променлива (closure).
    method, seed, episodes, eval_episodes, num_agents = job

    if method != "heuristic":
        # tag=f"seed{seed}" - секој паралелен job си пишува во СОПСТВЕНИ
        # results/{method}_seed{N}_* фајлови, без судир со останатите
        # job-ови кои го тренираат истиот метод со друг seed (видете
        # train.py::run_training коментар за tag).
        run_training(method=method, num_agents=num_agents, episodes=episodes, seed=seed, tag=f"seed{seed}")
        prefix = str(RESULTS_DIR / f"{method}_seed{seed}_model")
    else:
        prefix = None

    # EVAL_SEED_OFFSET (1 000 000) обезбедува дека евалуациските
    # seed-ови никогаш нема да се преклопат со тренинг seed-овите
    # (env.reset(seed=seed+ep) во train.py, за ep < episodes).
    summary = evaluate(
        method, prefix, num_agents=num_agents, episodes=eval_episodes,
        seed=EVAL_SEED_OFFSET + seed, tag=f"seed{seed}",
    )
    summary["seed"] = seed
    return method, summary


def main(episodes: int, eval_episodes: int, seeds: list[int], num_agents: int, max_workers: int | None):
    jobs = [(method, seed, episodes, eval_episodes, num_agents) for method in METHODS for seed in seeds]
    job_results = run_jobs(_run_one_job, jobs, max_workers=max_workers, label="метод×seed")

    all_results = {m: [] for m in METHODS}
    for method, summary in job_results:
        all_results[method].append(summary)
    for m in all_results:
        all_results[m].sort(key=lambda s: s["seed"])  # само за читлив/детерминистички ред во излезот, не влијае на статистиката

    # финална агрегирана табела: средина ± 95% доверителен интервал низ seed-овите
    print("\n\n===== АГРЕГИРАНИ РЕЗУЛТАТИ (средина низ seed-ови) =====")
    print(f"{'Метод':<15}{'Награда':<20}{'Стапка судири':<18}{'Стапка успех':<15}{'Просечни чекори':<18}")
    aggregate = {}
    for method, evals in all_results.items():
        rewards = [e["mean_reward"] for e in evals]
        coll = [e["collision_rate"] for e in evals]
        arr = [e["arrival_rate"] for e in evals]
        steps = [e["avg_steps"] for e in evals]

        def fmt(stats):
            return f"{stats['mean']:.2f} ± {stats['ci95']:.2f}"

        aggregate[method] = {
            "reward": agg_stats(rewards),
            "collision_rate": agg_stats(coll),
            "arrival_rate": agg_stats(arr),
            "avg_steps": agg_stats(steps),
        }
        row = aggregate[method]
        print(
            f"{method:<15}{fmt(row['reward']):<20}{fmt(row['collision_rate']):<18}"
            f"{fmt(row['arrival_rate']):<15}{fmt(row['avg_steps']):<18}"
        )

    out_path = RESULTS_DIR / "comparison_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"per_seed": all_results, "aggregate": aggregate}, f, indent=2, ensure_ascii=False)
    print(f"\nЦелосните резултати зачувани во: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500, help="тренинг епизоди по метод/seed")
    parser.add_argument("--eval_episodes", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--num_agents", type=int, default=3)
    parser.add_argument("--workers", type=int, default=None,
                         help="колку паралелни процеси (default: сите јадра - 2). --workers 1 = секвенцијално.")
    args = parser.parse_args()

    main(args.episodes, args.eval_episodes, args.seeds, args.num_agents, args.workers)
