"""
Observation-mode студија: ги тренира и евалуира Pixels-only и Fusion
режимите (IQL и VDN) низ повеќе seed-ови, паралелно (experiments/parallel.py,
исто како experiments/run_comparison.py), па прави ЕДНА обединета
obs_mode × метод табела заедно со веќе-готовите Kinematics резултати.

Ова е advanced ДОПОЛНИТЕЛНАТА студија - главната тема на трудот
(Kinematics × IQL/VDN/heuristic) веќе е завршена и статистички потврдена
преку experiments/run_comparison.py + experiments/significance_test.py
(results/comparison_summary.json). Kinematics×{iql,vdn} НЕ ги ре-тренирам
тука - ги реупотребувам директно оттаму (истиот seed-сет, ист
train/eval pipeline - kinematics режимот е byte-идентичен со порано, нема
методолошка разлика, само заштедено компјутерско време).

Употреба (dev, брзо):
    python experiments/observation_study.py --episodes 150 --eval_episodes 30 --seeds 0 1 2

Употреба (full, преку ноќ - исти episode/seed бројки како run_comparison.py):
    python experiments/observation_study.py --episodes 2000 --eval_episodes 200 --seeds 0 1 2 3 4 --workers 8

ВАЖНО за --workers: default (сите јадра - 2) МОЖЕ да ја надмине RAM на
машината - pixels/fusion replay buffer-ите чуваат слики (~100x поголеми
од kinematics по опсервација), па многу паралелни процеси кумулативно
бараат неколку GB секој (види PIXEL_BUFFER_SIZE подолу). --workers 8 е
тестирано безбедно на 16-јадрена/31GB машина - прилагоди спрема
сопствената RAM ако е поразлична.
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

from train import EVAL_SEED_OFFSET, RESULTS_DIR, run_training
from evaluate import evaluate
from experiments.parallel import agg_stats, run_jobs

METHODS = ["iql", "vdn"]
NEW_OBS_MODES = ["pixels", "fusion"]  # "kinematics" веќе постои (run_comparison.py)

# ВАЖНА РАЗЛИКА наспроти kinematics: default buffer_size=50_000
# (agents/networks.py::ReplayBuffer/MultiAgentReplayBuffer) складира
# СУРОВИ np.ndarray state-објекти - за kinematics тоа е ~168 бајти по
# опсервација (42 float32), но за pixels/fusion секоја слика е
# stack_size×H×W = 4×64×64 = 16 384 бајти (uint8), ~100x поголемо. На
# capacity=50_000 тоа значи неколку GB RAM ПО paralelen worker процес
# (state + next_state, ×num_agents за VDN) - со 14 паралелни job-ови
# (experiments/parallel.py::default_worker_count() на 16-јадрена машина)
# тоа лесно ја надминува вкупната RAM на машината (проверено: 31 GB
# вкупно). Затоа ГИ намалувам buffer_size/паралелизмот САМО за pixels/
# fusion job-овите - архитектурата/хиперпараметрите на самиот алгоритам
# (learning_rate, gamma, epsilon шема, target_update_every, episodes,
# seeds) остануваат ИДЕНТИЧНИ низ сите 3 режими; ова е чисто RAM-
# ограничување на replay buffer КАПАЦИТЕТОТ, не методолошка разлика во
# самото учење.
PIXEL_BUFFER_SIZE = 20_000


def _run_one_job(job):
    # top-level функција (не lambda/closure) - multiprocessing на Windows
    # користи "spawn", истата причина како во run_comparison.py::_run_one_job.
    method, obs_mode, seed, episodes, eval_episodes, num_agents = job
    # buffer_size=PIXEL_BUFFER_SIZE - RAM причина, види коментар кај
    # PIXEL_BUFFER_SIZE погоре (obs_mode тука е секогаш "pixels"/"fusion",
    # никогаш "kinematics" - NEW_OBS_MODES).
    run_training(method=method, num_agents=num_agents, episodes=episodes, seed=seed,
                 tag=f"seed{seed}", obs_mode=obs_mode, agent_kwargs={"buffer_size": PIXEL_BUFFER_SIZE})
    prefix = str(RESULTS_DIR / f"{method}_{obs_mode}_seed{seed}_model")
    summary = evaluate(
        method, prefix, num_agents=num_agents, episodes=eval_episodes,
        seed=EVAL_SEED_OFFSET + seed, tag=f"seed{seed}", obs_mode=obs_mode,
    )
    summary["seed"] = seed
    return obs_mode, method, summary


def _load_existing_kinematics(seeds: list[int]) -> dict[str, list[dict]]:
    """
    Ги вчитува веќе-готовите Kinematics (iql/vdn) резултати од
    results/comparison_summary.json (experiments/run_comparison.py), само
    за seed-овите кои реално се бараат тука - ако главната споредба била
    пуштена со ПОВЕЌЕ seed-ови, земам подмножество; ако со ПОМАЛКУ,
    ПРЕДУПРЕДУВАМ (не паѓа - вообичаено за "dev" рунди со помал --seeds
    сет од главната full студија).
    """
    path = RESULTS_DIR / "comparison_summary.json"
    if not path.exists():
        print(f"[observation_study] ПРЕДУПРЕДУВАЊЕ: {path} не постои - Kinematics редовите "
              f"нема да бидат во финалната табела (прво пушти experiments/run_comparison.py).")
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    out: dict[str, list[dict]] = {}
    for method in METHODS:
        evals = data["per_seed"].get(method, [])
        matching = [e for e in evals if e["seed"] in seeds]
        missing = set(seeds) - {e["seed"] for e in matching}
        if missing:
            print(f"[observation_study] ПРЕДУПРЕДУВАЊЕ: {method}/kinematics нема резултати "
                  f"за seed-ови {sorted(missing)} во {path}.")
        if matching:
            out[method] = matching
    return out


def main(episodes: int, eval_episodes: int, seeds: list[int], num_agents: int, max_workers: int | None):
    jobs = [
        (method, obs_mode, seed, episodes, eval_episodes, num_agents)
        for obs_mode in NEW_OBS_MODES
        for method in METHODS
        for seed in seeds
    ]
    job_results = run_jobs(_run_one_job, jobs, max_workers=max_workers, label="obs_mode×метод×seed")

    all_results: dict[tuple[str, str], list[dict]] = {(om, m): [] for om in NEW_OBS_MODES for m in METHODS}
    for obs_mode, method, summary in job_results:
        all_results[(obs_mode, method)].append(summary)
    for key in all_results:
        all_results[key].sort(key=lambda s: s["seed"])  # читлив/детерминистички ред, не влијае на статистиката

    for method, evals in _load_existing_kinematics(seeds).items():
        all_results[("kinematics", method)] = evals

    print("\n\n===== ОБЕДИНЕТА obs_mode × метод СПОРЕДБА (средина низ seed-ови) =====")
    print(f"{'obs_mode':<12}{'Метод':<10}{'Награда':<20}{'Стапка судири':<18}{'Стапка успех':<15}{'Просечни чекори':<18}")
    aggregate: dict[str, dict] = {}
    for (obs_mode, method), evals in all_results.items():
        if not evals:
            continue
        row = {
            "reward": agg_stats([e["mean_reward"] for e in evals]),
            "collision_rate": agg_stats([e["collision_rate"] for e in evals]),
            "arrival_rate": agg_stats([e["arrival_rate"] for e in evals]),
            "avg_steps": agg_stats([e["avg_steps"] for e in evals]),
        }
        aggregate.setdefault(obs_mode, {})[method] = row

        def fmt(stats):
            return f"{stats['mean']:.2f} ± {stats['ci95']:.2f}"

        print(
            f"{obs_mode:<12}{method:<10}{fmt(row['reward']):<20}{fmt(row['collision_rate']):<18}"
            f"{fmt(row['arrival_rate']):<15}{fmt(row['avg_steps']):<18}"
        )

    out_path = RESULTS_DIR / "observation_comparison.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "per_seed": {f"{om}/{m}": v for (om, m), v in all_results.items() if v},
                "aggregate": aggregate,
            },
            f, indent=2, ensure_ascii=False,
        )
    print(f"\nЦелосните резултати зачувани во: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=500, help="тренинг епизоди по (obs_mode, метод, seed)")
    parser.add_argument("--eval_episodes", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--num_agents", type=int, default=3)
    parser.add_argument("--workers", type=int, default=None,
                         help="колку паралелни процеси (default: сите јадра - 2). --workers 1 = секвенцијално.")
    args = parser.parse_args()

    main(args.episodes, args.eval_episodes, args.seeds, args.num_agents, args.workers)
