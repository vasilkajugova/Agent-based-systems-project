"""
Скалабилност: како се менуваат перформансите на секој метод кога го
менувам бројот контролирани возила (агенти) во сценариото?

Мотивација: целата главна споредба (run_comparison.py) е со фиксни 3
агенти. Но интересно прашање е дали заклучоците остануваат исти со
повеќе агенти - IQL по дефиниција треба да се влошува побрзо со бројот
агенти (секој агент гледа сè "понестационарна" околина колку повеќе
агенти учат истовремено), додека VDN (CTDE) треба да се справува подобро
токму затоа што тренирањето е централизирано. Ова директно го тестира
истражувачкото прашање на проектот, само низ дополнителна димензија
(num_agents) наместо со фиксна вредност.

Секој (метод, број-агенти, seed) е независен job, паралелизиран преку
experiments/parallel.py - истиот принцип како run_comparison.py.

Употреба (брз dev режим):
    python experiments/scalability_study.py --agent_counts 2 3 --episodes 100 --eval_episodes 20 --seeds 0 1

Употреба (целосен режим):
    python experiments/scalability_study.py --agent_counts 2 3 4 5 --episodes 1000 --eval_episodes 100 --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Кирилски print() на не-UTF8 конзола фрла UnicodeEncodeError - истата
# причина како во train.py, види коментар таму.
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
    method, seed, num_agents, episodes, eval_episodes = job
    # tag мора да го содржи И num_agents И seed - ако само seed, две
    # различни вредности на num_agents со ист seed би се судриле на исти
    # results/{method}_seed{N}_* патеки (види train.py::run_training).
    tag = f"scal_n{num_agents}_seed{seed}"

    if method != "heuristic":
        run_training(method=method, num_agents=num_agents, episodes=episodes, seed=seed, tag=tag)
        prefix = str(RESULTS_DIR / f"{method}_{tag}_model")
    else:
        prefix = None

    summary = evaluate(method, prefix, num_agents=num_agents, episodes=eval_episodes,
                        seed=EVAL_SEED_OFFSET + seed, tag=tag)
    summary["seed"] = seed
    summary["num_agents"] = num_agents
    return method, num_agents, summary


def main(agent_counts, episodes, eval_episodes, seeds, max_workers):
    jobs = [
        (method, seed, n, episodes, eval_episodes)
        for method in METHODS for n in agent_counts for seed in seeds
    ]
    job_results = run_jobs(_run_one_job, jobs, max_workers=max_workers, label="метод×N-агенти×seed")

    # ги организирам: {метод: {num_agents: [резултати по seed]}}
    per_run = {m: {n: [] for n in agent_counts} for m in METHODS}
    for method, n, summary in job_results:
        per_run[method][n].append(summary)

    print("\n\n===== СКАЛАБИЛНОСТ: средина низ seed-ови, по број агенти =====")
    aggregate = {m: {} for m in METHODS}
    for method in METHODS:
        print(f"\n{method}:")
        print(f"  {'N агенти':<10}{'Награда':<20}{'Стапка судири':<18}{'Стапка успех':<15}")
        for n in agent_counts:
            runs = per_run[method][n]
            row = {
                "reward": agg_stats([r["mean_reward"] for r in runs]),
                "collision_rate": agg_stats([r["collision_rate"] for r in runs]),
                "arrival_rate": agg_stats([r["arrival_rate"] for r in runs]),
                "avg_steps": agg_stats([r["avg_steps"] for r in runs]),
            }
            aggregate[method][str(n)] = row
            print(
                f"  {n:<10}"
                f"{row['reward']['mean']:.2f} ± {row['reward']['ci95']:.2f}    "
                f"{row['collision_rate']['mean']:.2f} ± {row['collision_rate']['ci95']:.2f}      "
                f"{row['arrival_rate']['mean']:.2f} ± {row['arrival_rate']['ci95']:.2f}"
            )

    # json.dump сепак ги претвора int клучевите во стрингови автоматски,
    # но правам експлицитно за да е јасно во самиот JSON фајл (и за
    # visualize_results.py да не се двоуми како да ги парсира клучевите)
    per_run_json = {m: {str(n): per_run[m][n] for n in agent_counts} for m in METHODS}

    out_path = RESULTS_DIR / "scalability_study.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"agent_counts": agent_counts, "per_run": per_run_json, "aggregate": aggregate}, f,
                   indent=2, ensure_ascii=False)
    print(f"\nРезултатите зачувани во: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_counts", type=int, nargs="+", default=[2, 3, 4, 5])
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--eval_episodes", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--workers", type=int, default=None, help="колку паралелни процеси (default: сите јадра - 2)")
    args = parser.parse_args()

    main(args.agent_counts, args.episodes, args.eval_episodes, args.seeds, args.workers)
