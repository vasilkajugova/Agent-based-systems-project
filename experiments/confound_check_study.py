"""
Контролна студија: го изолира ефектот на CTDE (VDN) наспроти независно
учење (IQL), одвоено од архитектурните разлики.
--------------------------------------------------------------------------
Мотивација (најдено при критичка ревизија на проектот): главната споредба
(experiments/run_comparison.py, results/comparison_summary.json) го
тренира IQL со `double_dqn=True, dueling=False`, а VDN со `dueling=True` и
БЕЗ double-DQN target воопшто (VDNAgent пред оваа ревизија немаше таа
опција - agents/vdn_agent.py секогаш користеше обичен `max()` target). Тоа
значи главниот "IQL vs VDN" резултат истовремено варира ДВА фактори:
  1. мешаat стратегија (независно учење наспроти CTDE decomposition) -
     ова е она што истражувачкото прашање тврди дека го тестира;
  2. архитектура/target-метод (dueling head, double-DQN target) - ова е
     конфаунд, не дел од истражувачкото прашање.

Оваа студија тренира ДВА методи каде точка 2 е ФИКСИРАНА иста за двата
(dueling=True, double_dqn=True за двата), а се разликува единствено точка
1 (VDN mixing наспроти целосно независни агенти):

  - "vdn_matched"    -> VDNAgent(dueling=True, double_dqn=True)   (нова double_dqn опција во agents/vdn_agent.py)
  - "iql_dueling"     -> IQLManager(dueling=True, double_dqn=True) (веќе постоеше во train.py::make_manager,
                          само никогаш не беше вклучен во главната споредба)

Ако наодот од главната студија (VDN тендира кон помала стапка на судири,
но статистички незначајно на n=5, results/significance_test.json) се
задржи и тука - тоа е посилен доказ дека разликата реално доаѓа од CTDE, не
од dueling/double-DQN. Ако наодот се промени (пр. стане значаен, или се
обрне насока) - тоа значи главната споредба навистина била под влијание
на архитектурниот конфаунд, и извештајот/трудот треба соодветно да се
преформулира.

Намерно ги користам ИСТИТЕ train.py::run_training/evaluate.py::evaluate
функции (не дуплирана train/eval петља, за разлика од
reward_ablation.py) - точно истиот pipeline како главната споредба
(experiments/run_comparison.py), само со различни agent_kwargs и tag, за
резултатите да се директно споредливи со results/comparison_summary.json
(истите episodes/eval_episodes/seeds → истите env поставки, само
архитектурата на VDN/IQL мрежите е сега порамнета).

Резултатски фајлови (НЕ ги допира постоечкиот results/comparison_summary.json):
  - results/vdn_matched_seed{N}_*        (модел/историја, преку train.py)
  - results/iql_dueling_matched_seed{N}_* (модел/историја, преку train.py)
  - results/confound_check_summary.json   (агрегирана табела, исто како comparison_summary.json)
  - results/confound_check_significance.json (Welch's t-test, исто како significance_test.py)

Употреба (брз "dev" режим - неколку минути):
    python experiments/confound_check_study.py --episodes 150 --eval_episodes 30 --seeds 0 1 2

Употреба (целосен режим, директно споредливо со главната табела):
    python experiments/confound_check_study.py --episodes 1500 --eval_episodes 150 --seeds 0 1 2 3 4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

sys.path.insert(0, str(Path(__file__).parent.parent))

from train import run_training, RESULTS_DIR, EVAL_SEED_OFFSET
from evaluate import evaluate
from experiments.parallel import run_jobs, agg_stats

# "matched" -> двата метода тука имаат dueling=True, double_dqn=True
# (единствената разлика е CTDE vs независно учење) - НЕ e истото како
# "iql"/"vdn" во run_comparison.py (таму IQL нема dueling, VDN нема
# double_dqn). Го гледам оваа студија како дополнување, не замена, на
# главната споредба.
METHODS = ["iql_dueling", "vdn"]
TAG_SUFFIX = "matched"


def _run_one_job(job):
    # top-level функција (multiprocessing на Windows користи "spawn",
    # истата причина како во run_comparison.py::_run_one_job)
    method, seed, episodes, eval_episodes, num_agents = job
    tag = f"{TAG_SUFFIX}_seed{seed}"

    agent_kwargs = {"double_dqn": True} if method == "vdn" else None
    run_training(method=method, num_agents=num_agents, episodes=episodes, seed=seed,
                 tag=tag, agent_kwargs=agent_kwargs)
    prefix = str(RESULTS_DIR / f"{method}_{tag}_model")

    summary = evaluate(
        method, prefix, num_agents=num_agents, episodes=eval_episodes,
        seed=EVAL_SEED_OFFSET + seed, tag=tag,
    )
    summary["seed"] = seed
    return method, summary


def main(episodes: int, eval_episodes: int, seeds: list[int], num_agents: int,
         max_workers: int | None, alpha: float):
    jobs = [(method, seed, episodes, eval_episodes, num_agents) for method in METHODS for seed in seeds]
    job_results = run_jobs(_run_one_job, jobs, max_workers=max_workers, label="метод×seed (matched)")

    all_results = {m: [] for m in METHODS}
    for method, summary in job_results:
        all_results[method].append(summary)
    for m in all_results:
        all_results[m].sort(key=lambda s: s["seed"])

    print("\n\n===== КОНТРОЛНА СТУДИЈА: dueling=True + double_dqn=True за ДВАТА метода =====")
    print(f"{'Метод':<18}{'Награда':<20}{'Стапка судири':<18}{'Стапка успех':<15}{'Просечни чекори':<18}")
    aggregate = {}
    for method, evals in all_results.items():
        metrics = {
            "reward": [e["mean_reward"] for e in evals],
            "collision_rate": [e["collision_rate"] for e in evals],
            "arrival_rate": [e["arrival_rate"] for e in evals],
            "avg_steps": [e["avg_steps"] for e in evals],
        }
        aggregate[method] = {k: agg_stats(v) for k, v in metrics.items()}
        row = aggregate[method]

        def fmt(s):
            return f"{s['mean']:.2f} ± {s['ci95']:.2f}"

        print(f"{method:<18}{fmt(row['reward']):<20}{fmt(row['collision_rate']):<18}"
              f"{fmt(row['arrival_rate']):<15}{fmt(row['avg_steps']):<18}")

    summary_path = RESULTS_DIR / "confound_check_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"per_seed": all_results, "aggregate": aggregate}, f, indent=2, ensure_ascii=False)
    print(f"\nЦелосните резултати зачувани во: {summary_path}")

    # Welch's t-test (истата логика/причина како во significance_test.py) -
    # директно врз "iql_dueling" vs "vdn" (двата matched) наместо врз
    # "iql" vs "vdn" (comparison_summary.json).
    print(f"\n===== Welch's t-test: iql_dueling vs vdn (matched, alpha={alpha}) =====")
    sig_results = []
    metric_keys = ["mean_reward", "collision_rate", "arrival_rate", "avg_steps"]
    a_evals, b_evals = all_results["iql_dueling"], all_results["vdn"]
    for metric in metric_keys:
        vals_a = [e[metric] for e in a_evals]
        vals_b = [e[metric] for e in b_evals]
        res = stats.ttest_ind(vals_a, vals_b, equal_var=False)
        significant = bool(res.pvalue < alpha)
        row = {
            "a": "iql_dueling", "b": "vdn", "metric": metric,
            "mean_a": sum(vals_a) / len(vals_a), "mean_b": sum(vals_b) / len(vals_b),
            "t": float(res.statistic), "df": float(res.df), "p": float(res.pvalue),
            "significant": significant,
        }
        sig_results.append(row)
        print(f"{metric:<18}{row['mean_a']:<12.4f}{row['mean_b']:<12.4f}"
              f"{row['t']:<10.3f}{row['df']:<8.2f}{row['p']:<10.4f}{'ДА' if significant else 'не'}")

    sig_path = RESULTS_DIR / "confound_check_significance.json"
    with open(sig_path, "w", encoding="utf-8") as f:
        json.dump({"alpha": alpha, "source": str(summary_path), "results": sig_results}, f, indent=2, ensure_ascii=False)
    print(f"Зачувано во: {sig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=500, help="тренинг епизоди по метод/seed")
    parser.add_argument("--eval_episodes", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--num_agents", type=int, default=3)
    parser.add_argument("--workers", type=int, default=None,
                         help="колку паралелни процеси (default: сите јадра - 2). --workers 1 = секвенцијално.")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    main(args.episodes, args.eval_episodes, args.seeds, args.num_agents, args.workers, args.alpha)
