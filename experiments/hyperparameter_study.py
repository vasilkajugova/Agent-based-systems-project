"""
Хиперпараметарска студија: дали default вредностите (learning_rate=1e-3,
discount_factor=0.95, epsilon target_fraction=0.7) се навистина разумен
избор, или некоја друга комбинација дава подобри резултати?

Пристап: варирам ЕДЕН хиперпараметар во исто време од baseline
конфигурацијата (one-at-a-time), не целосна grid-пребарување низ сите
комбинации - целосен grid (3×3×3=27 конфигурации × seeds) би бил
комбинаторно преголем за времето што го имам на располагање. One-at-a-time
е стандарден, разумен компромис за овој обем на проект - не гарантира
дека ја најдов ГЛОБАЛНО најдобрата комбинација (можни се interaction
ефекти меѓу хиперпараметрите), но дава разумна слика дали default
вредностите се во добар "регион", и кој хиперпараметар воопшто влијае
најмногу.

Само VDN (главниот метод во истражувачкото прашање) - истата логика важи
и за IQL, но времето за целосна студија на двата методи би се удвоило.

Употреба (брз dev режим):
    python experiments/hyperparameter_study.py --episodes 100 --eval_episodes 20 --seeds 0 1

Употреба (целосен режим):
    python experiments/hyperparameter_study.py --episodes 800 --eval_episodes 80 --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Кирилски print() на не-UTF8 конзола фрла UnicodeEncodeError - истата
# причина како во train.py, види коментар таму. (Ова е родителскиот
# процес - истата лекција штом еднаш ме "урна" преку ноќ на грчките букви
# во CONFIGS подолу, сега применета целосно и на кирилицата, не само на
# грчките букви.)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

sys.path.insert(0, str(Path(__file__).parent.parent))

from train import run_training, RESULTS_DIR, EVAL_SEED_OFFSET
from evaluate import evaluate
from experiments.parallel import run_jobs, agg_stats

METHOD = "vdn"

# тековните default вредности (agents/vdn_agent.py, train.py::compute_epsilon_decay)
BASELINE = {"learning_rate": 1e-3, "discount_factor": 0.95, "epsilon_target_fraction": 0.7}

# (slug за фајлови/tag-ови, читлива ознака за печатење, {параметар: нова вредност})
# секоја конфигурација е BASELINE со ТОЧНО ЕДЕН параметар сменет
CONFIGS = [
    # Намерно САМО ASCII знаци тука (без γ/ε) - ова буквално ми го урна
    # background run-от преку ноќ: Windows конзолата/redirect при паралелно
    # стартување не секогаш користи UTF-8 (кај мене испадна cp1251,
    # поддржува кирилица и "±", но НЕ и грчки букви), па print() на "γ=0.90"
    # фрлаше UnicodeEncodeError и целиот процес паѓаше. По оваа лекција,
    # секаде во проектот избегнувам не-ASCII знаци надвор од кирилицата.
    ("baseline", "baseline (lr=1e-3, gamma=0.95, eps_frac=0.7)", {}),
    ("lr_low", "lr=3e-4", {"learning_rate": 3e-4}),
    ("lr_high", "lr=3e-3", {"learning_rate": 3e-3}),
    ("gamma_low", "gamma=0.90", {"discount_factor": 0.90}),
    ("gamma_high", "gamma=0.99", {"discount_factor": 0.99}),
    ("eps_frac_low", "eps_frac=0.5", {"epsilon_target_fraction": 0.5}),
    ("eps_frac_high", "eps_frac=0.9", {"epsilon_target_fraction": 0.9}),
]


def _run_one_job(job):
    slug, overrides, seed, episodes, eval_episodes, num_agents = job
    cfg = {**BASELINE, **overrides}
    epsilon_target_fraction = cfg.pop("epsilon_target_fraction")
    agent_kwargs = cfg  # останува learning_rate, discount_factor - директно до VDNAgent конструкторот

    tag = f"hp_{slug}_seed{seed}"
    run_training(
        method=METHOD, num_agents=num_agents, episodes=episodes, seed=seed, tag=tag,
        epsilon_target_fraction=epsilon_target_fraction, agent_kwargs=agent_kwargs,
    )
    prefix = str(RESULTS_DIR / f"{METHOD}_{tag}_model")
    summary = evaluate(METHOD, prefix, num_agents=num_agents, episodes=eval_episodes,
                        seed=EVAL_SEED_OFFSET + seed, tag=tag)
    summary["seed"] = seed
    return slug, summary


def main(episodes, eval_episodes, seeds, num_agents, max_workers):
    jobs = [
        (slug, overrides, seed, episodes, eval_episodes, num_agents)
        for slug, label, overrides in CONFIGS for seed in seeds
    ]
    job_results = run_jobs(_run_one_job, jobs, max_workers=max_workers, label="конфигурација×seed")

    per_run = {slug: [] for slug, label, overrides in CONFIGS}
    for slug, summary in job_results:
        per_run[slug].append(summary)

    print("\n\n===== ХИПЕРПАРАМЕТАРСКА СТУДИЈА (VDN): средина низ seed-ови =====")
    print(f"{'Конфигурација':<32}{'Награда':<20}{'Стапка судири':<18}{'Стапка успех':<15}")
    aggregate = {}
    for slug, label, overrides in CONFIGS:
        runs = per_run[slug]
        row = {
            "reward": agg_stats([r["mean_reward"] for r in runs]),
            "collision_rate": agg_stats([r["collision_rate"] for r in runs]),
            "arrival_rate": agg_stats([r["arrival_rate"] for r in runs]),
            "avg_steps": agg_stats([r["avg_steps"] for r in runs]),
        }
        aggregate[slug] = {"label": label, **row}
        print(
            f"{label:<32}"
            f"{row['reward']['mean']:.2f} ± {row['reward']['ci95']:.2f}    "
            f"{row['collision_rate']['mean']:.2f} ± {row['collision_rate']['ci95']:.2f}      "
            f"{row['arrival_rate']['mean']:.2f} ± {row['arrival_rate']['ci95']:.2f}"
        )

    out_path = RESULTS_DIR / "hyperparameter_study.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"baseline": BASELINE, "per_run": per_run, "aggregate": aggregate}, f,
                   indent=2, ensure_ascii=False)
    print(f"\nРезултатите зачувани во: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--eval_episodes", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--num_agents", type=int, default=3)
    parser.add_argument("--workers", type=int, default=None, help="колку паралелни процеси (default: сите јадра - 2)")
    args = parser.parse_args()

    main(args.episodes, args.eval_episodes, args.seeds, args.num_agents, args.workers)
