"""
Robustness студија - advanced дополнителна студија врз ВЕЌЕ истренираните
Kinematics/Pixels/Fusion модели (IQL и VDN, `experiments/observation_study.py`
+ постоечкиот `experiments/run_comparison.py`). За секој (метод, obs_mode,
seed) веќе-истрениран модел: clean евалуација + секој ПРИМЕНЛИВ perturbation
услов (experiments/robustness.py::APPLICABLE_CONDITIONS) - шум на
kinematics, blur/darken на pixel сликата, привремено гасење на една
observation-гранка (само Fusion).

Методолошки важно: моделите НЕ се re-тренираат тука - истиот модел (исти
тежини) се евалуира еднаш "чист" и еднаш под секој perturbation, за да
падот во перформанси ДИРЕКТНО ја покажува робустноста на веќе-научената
политика на несовршени сензори, не некаков различен тренинг сетинг.

За секој (метод, obs_mode) двојка, "падот" (%) го пресметувам ПАРИРАНО по
seed (clean_seed vs perturbed_seed на ИСТИОТ seed/модел), не просто
разлика на веќе-агрегирани средини - паметно ги отстранува варијациите
"овој seed случајно излезе подобар модел" од самата споредба
clean-vs-perturbed.

Употреба:
    python experiments/robustness_study.py --seeds 0 1 2 3 4 --eval_episodes 150
    python experiments/robustness_study.py --methods vdn --obs_modes fusion --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Кирилски print() на не-UTF8 конзола фрла UnicodeEncodeError - истата
# причина како во train.py, види коментар таму.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

sys.path.insert(0, str(Path(__file__).parent.parent))

from train import EVAL_SEED_OFFSET, RESULTS_DIR
from evaluate import evaluate
from experiments.parallel import agg_stats, run_jobs
from experiments.robustness import APPLICABLE_CONDITIONS, make_perturb_fn

METRICS = ["mean_reward", "collision_rate", "arrival_rate", "avg_steps"]
# Насока на "влошување" по метрика - множител применет на суровиот %
# промена (cond vs clean) така што РЕЗУЛТАТОТ секогаш е позитивен кога
# состојбата е ПОЛОША од clean: reward/arrival_rate се влошуваат кога
# ОПАЃААТ (множител -1), collision_rate се влошува кога РАСТЕ (множител
# +1). avg_steps нема еднозначна "подобро/полошо" насока (подолга епизода
# може да значи и повеќе претпазливост И повеќе бесцелно "стоење и
# чекање") - го известувам како СУРОВ % промена (множител +1), чисто
# описно, без имплицитно судење.
DIRECTION = {"mean_reward": -1, "collision_rate": 1, "arrival_rate": -1, "avg_steps": 1}


def find_model_prefix(method: str, obs_mode: str, seed: int) -> str | None:
    """
    Го наоѓа зачуваниот модел за (метод, obs_mode, seed) - прво tag-ираниот
    results/{method}[_{obs_mode}]_seed{N}_model (train.py::run_training
    именување, види таму), потоа "голиот" (директно `python train.py`, без
    --tag). Ако ниту едно не постои, враќа None - job-от за тој (метод,
    obs_mode, seed) едноставно се прескокнува (моделот сè уште не е
    истрениран).
    """
    mode_suffix = "" if obs_mode == "kinematics" else f"_{obs_mode}"
    tagged = RESULTS_DIR / f"{method}{mode_suffix}_seed{seed}_model"
    if (RESULTS_DIR / f"{method}{mode_suffix}_seed{seed}_model_agent0.pt").exists():
        return str(tagged)
    bare = RESULTS_DIR / f"{method}{mode_suffix}_model"
    if (RESULTS_DIR / f"{method}{mode_suffix}_model_agent0.pt").exists():
        return str(bare)
    return None


def _run_one_job(job):
    # top-level функција (не lambda/closure) - multiprocessing на Windows
    # користи "spawn", истата причина како во run_comparison.py::_run_one_job.
    method, obs_mode, seed, eval_episodes, num_agents, strengths = job
    model_prefix = find_model_prefix(method, obs_mode, seed)
    if model_prefix is None:
        return method, obs_mode, seed, None

    conditions = ["clean"] + APPLICABLE_CONDITIONS[obs_mode]
    eval_seed = EVAL_SEED_OFFSET + seed
    results = {}
    for cond in conditions:
        perturb_fn = None if cond == "clean" else make_perturb_fn(cond, strengths.get(cond))
        summary = evaluate(
            method, model_prefix, num_agents=num_agents, episodes=eval_episodes,
            seed=eval_seed, save=False, obs_mode=obs_mode, perturb_fn=perturb_fn,
        )
        results[cond] = summary
    return method, obs_mode, seed, results


def _pct_change(clean: float, cond: float) -> float:
    if clean == 0:
        return 0.0
    return (cond - clean) / abs(clean) * 100.0


def main(methods, obs_modes, seeds, eval_episodes, num_agents, strengths, max_workers):
    jobs = [
        (method, obs_mode, seed, eval_episodes, num_agents, strengths)
        for method in methods
        for obs_mode in obs_modes
        for seed in seeds
    ]
    job_results = run_jobs(_run_one_job, jobs, max_workers=max_workers, label="метод×obs_mode×seed")

    # raw[(method, obs_mode)][condition] = листа summary-и, по еден на seed (по редослед на seeds)
    raw: dict[tuple[str, str], dict[str, list]] = {}
    skipped = []
    for method, obs_mode, seed, results in job_results:
        if results is None:
            skipped.append((method, obs_mode, seed))
            continue
        key = (method, obs_mode)
        raw.setdefault(key, {})
        for cond, summary in results.items():
            raw[key].setdefault(cond, []).append(summary)

    if skipped:
        print(f"\n[robustness] прескокнати (нема зачуван модел): {skipped}")

    aggregate = {}
    print("\n===== Robustness студија (пад % наспроти clean, парирано по seed) =====")
    for (method, obs_mode), by_cond in raw.items():
        clean_evals = by_cond.get("clean")
        if not clean_evals:
            continue
        key_str = f"{method}/{obs_mode}"
        aggregate[key_str] = {"clean": {m: agg_stats([e[m] for e in clean_evals]) for m in METRICS}}
        print(f"\n--- {key_str} ({len(clean_evals)} seed-ови) ---")
        print(f"{'услов':<14}{'reward':<22}{'collision':<20}{'arrival':<20}{'steps':<18}")
        clean_row = aggregate[key_str]["clean"]
        print(
            f"{'clean':<14}"
            f"{clean_row['mean_reward']['mean']:.2f}±{clean_row['mean_reward']['ci95']:.2f}".ljust(22)
            + f"{clean_row['collision_rate']['mean']:.2f}±{clean_row['collision_rate']['ci95']:.2f}".ljust(20)
            + f"{clean_row['arrival_rate']['mean']:.2f}±{clean_row['arrival_rate']['ci95']:.2f}".ljust(20)
            + f"{clean_row['avg_steps']['mean']:.1f}±{clean_row['avg_steps']['ci95']:.1f}"
        )

        for cond in APPLICABLE_CONDITIONS[obs_mode]:
            cond_evals = by_cond.get(cond)
            if not cond_evals:
                continue
            # ПАРИРАНО по seed: clean_evals[i] и cond_evals[i] доаѓаат од ИСТ
            # seed (истиот job ги генерира двете последователно, во истиот
            # seed-редослед - види _run_one_job), значи индексите се усогласени.
            metric_stats = {m: agg_stats([e[m] for e in cond_evals]) for m in METRICS}
            # позитивен % = ПОЛОШО од clean (види DIRECTION погоре) - парирано
            # по seed (clean_evals[i]/cond_evals[i] се ИСТИОТ seed/модел).
            pct_drops = {
                m: agg_stats([DIRECTION[m] * _pct_change(c[m], p[m]) for c, p in zip(clean_evals, cond_evals)])
                for m in METRICS
            }
            aggregate[key_str][cond] = {"metrics": metric_stats, "pct_change_vs_clean": pct_drops}
            print(
                f"{cond:<14}"
                f"{metric_stats['mean_reward']['mean']:.2f}±{metric_stats['mean_reward']['ci95']:.2f} "
                f"({pct_drops['mean_reward']['mean']:+.0f}%)".ljust(22)
                + f"{metric_stats['collision_rate']['mean']:.2f} ({pct_drops['collision_rate']['mean']:+.0f}%)".ljust(20)
                + f"{metric_stats['arrival_rate']['mean']:.2f} ({pct_drops['arrival_rate']['mean']:+.0f}%)".ljust(20)
                + f"{metric_stats['avg_steps']['mean']:.1f} ({pct_drops['avg_steps']['mean']:+.0f}%)"
            )

    out_path = RESULTS_DIR / "robustness_study.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "strengths": strengths,
                "eval_episodes": eval_episodes,
                "seeds": seeds,
                "skipped": skipped,
                "aggregate": aggregate,
            },
            f, indent=2, ensure_ascii=False,
        )
    print(f"\nЗачувано во: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", default=["iql", "vdn"], choices=["iql", "vdn"])
    parser.add_argument("--obs_modes", nargs="+", default=["kinematics", "pixels", "fusion"],
                         choices=["kinematics", "pixels", "fusion"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--eval_episodes", type=int, default=150)
    parser.add_argument("--num_agents", type=int, default=3)
    parser.add_argument("--kin_noise_sigma", type=float, default=0.15,
                         help="Gaussian шум sigma на нормализирани ([-1,1]) kinematics колони")
    parser.add_argument("--pixel_blur_sigma", type=float, default=2.0, help="Gaussian blur sigma (пиксели)")
    parser.add_argument("--pixel_dark_factor", type=float, default=0.25, help="множител на интензитет (< 1 = потемно)")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    strengths = {
        "kin_noise": args.kin_noise_sigma,
        "pixel_blur": args.pixel_blur_sigma,
        "pixel_dark": args.pixel_dark_factor,
        "drop_kin": None,
        "drop_img": None,
    }
    main(args.methods, args.obs_modes, args.seeds, args.eval_episodes, args.num_agents, strengths, args.workers)
