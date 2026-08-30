"""
Q-вредностна чувствителност на Fusion-мрежите на секоја гранка (kin/img) -
ја формализира/зачувува "modality collapse" дијагностиката што порано ја
правев ad-hoc (не беше зачувана во репозиториумот) а сепак е цитирана со
конкретни бројки во README.md/README.en.md.

Идеја: за секоја веќе-истренирана Fusion мрежа (agent i, seed s), земам
реални on-policy состојби (тековната политика, epsilon=0, преку неколку
епизоди во env-от), и за секоја состојба ги споредувам:
    Q_full      = Q(kin, img)                     - нормална опсервација
    Q_drop_img  = Q(kin, ZERO)                     - agents/networks.py::drop_branch(state,"img")
    Q_drop_kin  = Q(ZERO, img)                     - agents/networks.py::drop_branch(state,"kin")
img_sens/kin_sens = средна апсолутна промена на Q-вредностите (по акција,
па просек) кога соодветната ГРАНКА Е ИСКЛУЧЕНА. Ако мрежата реално ја
користи сликата за да одлучува, гасењето на сликата треба значително да
ги смени Q-вредностите (img_sens голем); ако ја игнорира ("collapse"),
Q-вредностите остануваат речиси исти дури и со целосно црна слика
(img_sens ≈ 0).

Ова НЕ е episode-level метрика (за тоа види robustness_study.py -
drop_img/drop_kin условите таму, кои мерат надолна низа низ цела
епизода/env rollout). Оваа метрика е "суров" сигнал директно на
Q-мрежата, без да минува low дискретизацијата на argmax+env dynamics,
па е поостра/поточна дијагностика за самиот modality collapse феномен.

Употреба:
    python experiments/modality_sensitivity.py --obs_modes fusion fusion_fixed --seeds 0 1 2 3 4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.iql_manager import IQLManager
from agents.vdn_agent import VDNAgent
from agents.networks import drop_branch, forward_q, to_batch_tensor
from envs.multi_agent_intersection import MultiAgentIntersectionEnv
from train import EVAL_SEED_OFFSET, RESULTS_DIR
from experiments.parallel import agg_stats

METHODS = ["iql", "vdn"]
# име на резултатски префикс по obs_mode (совпаѓа со train.py::mode_suffix
# конвенцијата - "fusion" -> "{method}_fusion_seed{s}_model",
# "fusion_fixed" -> "{method}_fusion_fixed_seed{s}_model", види
# experiments/observation_study.py и experiments/fusion_fix_study.py).
PREFIX_TAG = {"fusion": "fusion", "fusion_fixed": "fusion_fixed"}


def _load_manager(method: str, num_agents: int, obs_dim: int, num_actions: int, img_shape, prefix: str):
    if method == "iql":
        m = IQLManager(num_agents, obs_dim, num_actions, double_dqn=True, dueling=False,
                        obs_mode="fusion", img_shape=img_shape)
    else:
        m = VDNAgent(num_agents, obs_dim, num_actions, dueling=True,
                      obs_mode="fusion", img_shape=img_shape)
    m.load(prefix)
    return m


def _agent_model(manager, method: str, i: int):
    return manager.agents[i].model if method == "iql" else manager.models[i]


def _collect_states(env, manager, method: str, num_agents: int, episodes: int, seed: int):
    """On-policy состојби (epsilon=0) од `episodes` евалуациони епизоди, по агент."""
    per_agent_states = [[] for _ in range(num_agents)]
    for ep in range(episodes):
        states = env.reset(seed=EVAL_SEED_OFFSET + seed + ep * 1000)
        done_flags = [False] * num_agents
        steps = 0
        while not all(done_flags) and steps < 30:
            actions = manager.get_actions(states, epsilon=0.0)
            for i, s in enumerate(states):
                if not done_flags[i]:
                    per_agent_states[i].append(s)
            states, rewards, dones, info = env.step(actions)
            done_flags = [d or done_flags[i] for i, d in enumerate(dones)]
            steps += 1
    return per_agent_states


def _sensitivity_for_agent(model, states, device):
    if not states:
        return float("nan"), float("nan")
    img_diffs, kin_diffs = [], []
    with torch.no_grad():
        for s in states:
            q_full = forward_q(model, to_batch_tensor(s, device))
            q_drop_img = forward_q(model, to_batch_tensor(drop_branch(s, "img"), device))
            q_drop_kin = forward_q(model, to_batch_tensor(drop_branch(s, "kin"), device))
            img_diffs.append(float(torch.abs(q_full - q_drop_img).mean().item()))
            kin_diffs.append(float(torch.abs(q_full - q_drop_kin).mean().item()))
    return float(np.mean(img_diffs)), float(np.mean(kin_diffs))


def main(obs_modes: list[str], methods: list[str], seeds: list[int], num_agents: int, episodes: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = MultiAgentIntersectionEnv(num_agents=num_agents, obs_mode="fusion")
    obs_dim, img_shape, num_actions = env.obs_dim, env.img_shape, env.num_actions

    per_combo: dict[str, dict[str, list[float]]] = {}
    try:
        for obs_mode in obs_modes:
            tag = PREFIX_TAG[obs_mode]
            for method in methods:
                img_vals, kin_vals = [], []
                for seed in seeds:
                    prefix = str(RESULTS_DIR / f"{method}_{tag}_seed{seed}_model")
                    if not Path(f"{prefix}_agent0.pt").exists():
                        print(f"[modality_sensitivity] прескокнувам {method}/{obs_mode} seed{seed} - "
                              f"{prefix}_agent0.pt не постои")
                        continue
                    manager = _load_manager(method, num_agents, obs_dim, num_actions, img_shape, prefix)
                    per_agent_states = _collect_states(env, manager, method, num_agents, episodes, seed)
                    for i in range(num_agents):
                        model = _agent_model(manager, method, i)
                        img_s, kin_s = _sensitivity_for_agent(model, per_agent_states[i], device)
                        img_vals.append(img_s)
                        kin_vals.append(kin_s)
                        print(f"{method}/{obs_mode} seed{seed} agent{i}: img_sens={img_s:.4f} kin_sens={kin_s:.4f}")
                per_combo[f"{obs_mode}/{method}"] = {"img_sens": img_vals, "kin_sens": kin_vals}
    finally:
        env.close()

    print("\n\n===== img_sens / kin_sens агрегат (низ seed×агент) =====")
    print(f"{'obs_mode/метод':<22}{'img_sens':<28}{'kin_sens':<28}{'ratio (kin/img)':<15}")
    summary = {}
    for key, vals in per_combo.items():
        if not vals["img_sens"]:
            continue
        img_stats = agg_stats(vals["img_sens"])
        kin_stats = agg_stats(vals["kin_sens"])
        ratio = (kin_stats["mean"] / img_stats["mean"]) if img_stats["mean"] > 1e-12 else float("inf")
        summary[key] = {"img_sens": img_stats, "kin_sens": kin_stats, "ratio_kin_over_img": ratio}
        print(f"{key:<22}{img_stats['mean']:.4f} ± {img_stats['ci95']:.4f}      "
              f"{kin_stats['mean']:.4f} ± {kin_stats['ci95']:.4f}      {ratio:.1f}x")

    out_path = RESULTS_DIR / "modality_sensitivity.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"per_combo_raw": per_combo, "summary": summary,
                    "seeds": seeds, "episodes_per_seed": episodes}, f, indent=2, ensure_ascii=False)
    print(f"\nЗачувано во: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obs_modes", nargs="+", default=["fusion", "fusion_fixed"],
                         choices=["fusion", "fusion_fixed"])
    parser.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--num_agents", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=20, help="on-policy евалуациони епизоди по seed за собирање состојби")
    args = parser.parse_args()

    main(args.obs_modes, args.methods, args.seeds, args.num_agents, args.episodes)
