"""
Евалуација и статистичка споредба на веќе истренираните агенти (или на
heuristic baseline) низ повеќе seed-ови. Ова ми е потребно за да имам
строга евалуација - еден единствен резултат не кажува ништо (можел да е
"среќа"), затоа секогаш пуштам повеќе seed-ови и пресметувам доверителен
интервал.

Метрики што ги пресметувам по епизода:
  - mean_reward: просечна награда по агент
  - collision_rate: дел епизоди со барем еден судир (безбедност)
  - arrival_rate: дел агенти кои успешно стигнале до дестинација (успешност)
  - avg_steps: просечна должина на епизода (пропусна моќ/ефикасност)

Забелешка за seed-от: default вредноста тука е EVAL_SEED_OFFSET
(увезена од train.py) - истиот "безбеден" офсет што го користам и во
experiments/run_comparison.py и experiments/reward_ablation.py. Ова го
додадов откако сфатив дека ако некој ја стартува оваа скрипта директно
(без да оди преку run_comparison.py) по тренинг со многу епизоди, постои
ризик евалуацијата да користи seed кој веќе бил "видени" при тренирање -
истиот проблем со преклопување seed-ови (data leakage) што веќе го имав
решено на друго место во проектот, само овде го немав применето.

Употреба:
    python evaluate.py --method vdn --model_prefix results/vdn_model --episodes 50
    python evaluate.py --method vdn --episodes 50 --obs_mode pixels   # advanced студија, default prefix се менува соодветно
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from envs.multi_agent_intersection import MultiAgentIntersectionEnv
from agents.iql_manager import IQLManager
from agents.vdn_agent import VDNAgent
from agents.heuristic_agent import HeuristicAgent
from train import EVAL_SEED_OFFSET, IDLE_ACTION

# Кирилски print() на не-UTF8 конзола фрла UnicodeEncodeError - истата
# причина како во train.py, види коментар таму.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

RESULTS_DIR = Path(__file__).parent / "results"


def load_manager(
    method: str,
    model_prefix: str | None,
    num_agents: int,
    obs_dim: int,
    num_actions: int,
    obs_mode: str = "kinematics",
    img_shape: tuple[int, int, int] | None = None,
):
    # исто како make_manager() во train.py, само тука дополнително и го вчитувам зачуваниот модел
    obs_kwargs = dict(obs_mode=obs_mode, img_shape=img_shape)
    if method == "iql":
        m = IQLManager(num_agents, obs_dim, num_actions, double_dqn=True, **obs_kwargs)
        m.load(model_prefix)
        return m
    if method == "iql_dueling":
        m = IQLManager(num_agents, obs_dim, num_actions, double_dqn=True, dueling=True, **obs_kwargs)
        m.load(model_prefix)
        return m
    if method == "vdn":
        m = VDNAgent(num_agents, obs_dim, num_actions, dueling=True, **obs_kwargs)
        m.load(model_prefix)
        return m
    if method == "heuristic":
        return HeuristicAgent()  # нема модел за вчитување, heuristic не учи
    raise ValueError(method)


def evaluate(method: str, model_prefix: str | None, num_agents: int = 3, episodes: int = 50,
             seed: int = EVAL_SEED_OFFSET, tag: str | None = None, save: bool = True,
             obs_mode: str = "kinematics", perturb_fn=None):
    """
    perturb_fn (опционално) - функција states -> states (листа по агент)
    која се применува ВЕДНАШ ПРЕД manager.get_actions(...), на секој чекор.
    Го користи experiments/robustness_study.py за да ги "расипе" опсервациите
    (шум/blur/darken/branch-dropout) без да ја дуплира целата eval-логика
    тука - самиот модел е ВЕЌЕ истрениран на чисти опсервации (тренингот
    воопшто не знае за perturb_fn), значи ова е чисто eval-time
    манипулација, не дел од МDP-то со кое агентот учел.
    """
    env = MultiAgentIntersectionEnv(num_agents=num_agents, obs_mode=obs_mode)
    try:
        manager = load_manager(method, model_prefix, num_agents, env.obs_dim, env.num_actions,
                                obs_mode=obs_mode, img_shape=env.img_shape)

        per_episode = []
        for ep in range(episodes):
            states = env.reset(seed=seed + ep)
            ep_reward = np.zeros(num_agents)
            collided = False
            arrivals = 0
            steps = 0
            done_all = False
            # ИСТАТА "замрзни го завршениот агент на IDLE" логика како во
            # train.py::run_training() - претходно овој loop ја немаше (само
            # train.py ја имаше), инконзистентност train/eval: без ова, агент
            # кој веќе пристигнал/се судрил сепак добива нова акција од
            # политиката на секој нареден чекор додека ги чека тимските
            # другари, наместо едноставно да "стои". Наградата веќе не се
            # "истекува" по поправката во envs/multi_agent_intersection.py
            # (секогаш е 0 за веќе-завршен агент), но акцијата сепак треба да
            # биде конзистентна со train.py заради коректност/читливост.
            active = np.ones(num_agents, dtype=bool)
            while not done_all:
                # perturb_fn се применува само на КОПИЈАТА што ја гледа
                # политиката (obs_for_policy) - вистинската `states` (и се што
                # произлегува од env.step(), пр. crashed/arrived) остануваат
                # непроменети. Со други зборови: го "расипувам" сетилото на
                # агентот, не самата симулација/динамика.
                obs_for_policy = perturb_fn(states) if perturb_fn is not None else states
                if method == "heuristic":
                    actions = manager.get_actions(obs_for_policy)
                else:
                    actions = manager.get_actions(obs_for_policy, epsilon=0.0)  # epsilon=0 значи чиста политика, БЕЗ случајно истражување
                actions = [a if active[i] else IDLE_ACTION for i, a in enumerate(actions)]
                states, rewards, dones, info = env.step(actions)
                ep_reward += np.array(rewards)
                collided = collided or any(info["crashed"])
                arrivals = sum(info["arrived"])
                active = active & ~np.array(dones)
                steps += 1
                done_all = all(dones)

            per_episode.append(
                {
                    "episode": ep,
                    "mean_reward": float(ep_reward.mean()),
                    "collided": collided,
                    "arrivals": arrivals,
                    "steps": steps,
                }
            )

    finally:
        env.close()

    rewards = [e["mean_reward"] for e in per_episode]
    collision_rate = np.mean([e["collided"] for e in per_episode])
    arrival_rate = np.mean([e["arrivals"] for e in per_episode]) / num_agents
    avg_steps = np.mean([e["steps"] for e in per_episode])

    summary = {
        "method": method,
        "obs_mode": obs_mode,
        "episodes": episodes,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "ci95_reward": float(1.96 * np.std(rewards) / np.sqrt(episodes)),  # 95% доверителен интервал
        "collision_rate": float(collision_rate),
        "arrival_rate": float(arrival_rate),
        "avg_steps": float(avg_steps),
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    # save=False / tag го користам од experiments/parallel.py-базираните
    # скрипти (run_comparison.py, scalability_study.py,
    # hyperparameter_study.py) - ако повеќе паралелни процеси го евалуираат
    # ИСТИОТ метод (различни seed-ови/конфигурации) без tag, сите би
    # пишувале во ИСТ results/{method}_eval.json истовремено (race
    # condition, на Windows дури и PermissionError). Секој повикувач сепак
    # веќе го добива summary-то во меморија преку return вредноста и си ги
    # прави сопствените агрегирани резултати - овој фајл е само пригодност
    # за директно (не-паралелно) стартување.
    #
    # mode_suffix - исто именување како во train.py::run_training(): празно
    # за kinematics (backward-compat), "_pixels"/"_fusion" инаку.
    if save:
        mode_suffix = "" if obs_mode == "kinematics" else f"_{obs_mode}"
        name = f"{method}{mode_suffix}" if tag is None else f"{method}{mode_suffix}_{tag}"
        out_path = RESULTS_DIR / f"{name}_eval.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "episodes": per_episode}, f, indent=2)
        print(f"Евалуацијата зачувана во: {out_path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["iql", "iql_dueling", "vdn", "heuristic"], required=True)
    parser.add_argument("--model_prefix", default=None, help="патека до зачуван модел (results/<method>_model)")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--num_agents", type=int, default=3)
    parser.add_argument("--seed", type=int, default=EVAL_SEED_OFFSET,
                         help="почетен seed за евалуационите епизоди (default: EVAL_SEED_OFFSET, "
                              "исто како во run_comparison.py/reward_ablation.py - не се преклопува "
                              "со тренинг seed-овите ни при многу долги тренинзи)")
    parser.add_argument("--obs_mode", choices=["kinematics", "pixels", "fusion"], default="kinematics")
    args = parser.parse_args()

    mode_suffix = "" if args.obs_mode == "kinematics" else f"_{args.obs_mode}"
    prefix = args.model_prefix or str(RESULTS_DIR / f"{args.method}{mode_suffix}_model")
    evaluate(args.method, prefix if args.method != "heuristic" else None, args.num_agents, args.episodes,
             seed=args.seed, obs_mode=args.obs_mode)
