"""
Ablation студија #1: тежина на collision_reward
------------------------------------------------
Мотивација: сакам да проверам дали построга казна за судир (collision_reward
понегативен од default -5) навистина ја намалува стапката на судири, или
агентот "профитира" со брзо возење и покрај повремени судири, бидејќи
наградата за брзина ја надополнува загубата од казната.

(Забелешка додадена подоцна, при ревизија на кодот: претходна верзија на
овој коментар тврдеше дека IQL/VDN имаат ПОВИСОКА стапка на судири од
heuristic baseline-от (0.64-0.66 наспроти 0.50) - тоа веќе не важи.
Точните тековни бројки секогаш ги гледам во свежо-регенерираниот
`results/comparison_summary.json` (намерно не ги преземам тука во
коментар - веќе еднаш застарија по поправка во кодот, па конкретни
бројки во коментар се "кревки"; JSON-от е единствениот извор на
вистина). Квалитативно, по неколку поправки во меѓувреме (VDN
target-masking во agents/vdn_agent.py, поправката на "истечената"
награда во envs/multi_agent_intersection.py), VDN и IQL веќе не се
"полоши" од heuristic по безбедност. Го оставам овој коментар тука
наместо тивко да го избришам - самата ablation студија (варирање на
collision_reward) си останува валидна и интересна независно од
првичната мотивација.)

Оваа ablation студија тренира VDN со неколку различни вредности за
collision_reward и мери дали построгата казна навистина ја подобрува
безбедноста (и по која цена во arrival_rate/reward - можеби агентот
станува премногу "плашлив" и не стигнува никаде).

Ablation студија #2: courtesy (fairness) shaping член
------------------------------------------------------
Тренира VDN со courtesy_weight=0 (без мојот shaping член - baseline)
наспроти courtesy_weight>0 (мојот оригинален придонес) за да измерам
дали агентите стануваат "пофер" кон human-driven возилата (помалку
агресивно возење во нивна близина), и по која цена во ефикасност.

И двете студии ги пуштам паралелно преку experiments/parallel.py -
секоја (вредност, seed) комбинација е независен job (train_and_eval_vdn
не пишува никакви фајлови, само враќа резултат во меморија), па нема
причина да чекаат едно по друго.

Употреба (брз dev режим):
    python experiments/reward_ablation.py --episodes 100 --eval_episodes 20 --seeds 0 1

Употреба (целосен режим за финалните резултати):
    python experiments/reward_ablation.py --episodes 1000 --eval_episodes 100 --seeds 0 1 2 3 4
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

from envs.multi_agent_intersection import MultiAgentIntersectionEnv
from agents.vdn_agent import VDNAgent
from train import compute_epsilon_decay, set_global_seed, EVAL_SEED_OFFSET, IDLE_ACTION  # ги користам истите epsilon-шема/seeding/masking како во главниот train.py
from experiments.parallel import run_jobs, agg_stats

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def train_and_eval_vdn(
    config_overrides: dict,
    courtesy_weight: float,
    num_agents: int,
    train_episodes: int,
    eval_episodes: int,
    seed: int,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: float | None = None,
    target_update_every: int = 10,
):
    set_global_seed(seed)
    env = MultiAgentIntersectionEnv(
        num_agents=num_agents,
        config_overrides=config_overrides,
        courtesy_weight=courtesy_weight,
    )
    agent = VDNAgent(num_agents, env.obs_dim, env.num_actions, dueling=True)

    # ИСТАТА автоматска epsilon-скалирачка формула како во train.py - ова
    # ми е важно за да можам да ги споредам резултатите од оваа ablation
    # студија со главната споредба (run_comparison.py) без методолошка
    # разлика во exploration шемата да ми ги искриви заклучоците.
    if epsilon_decay is None:
        epsilon_decay = compute_epsilon_decay(train_episodes, epsilon_start, epsilon_min)

    # ---- тренинг ----
    epsilon = epsilon_start
    for ep in range(train_episodes):
        states = env.reset(seed=seed + ep)
        done_all = False
        # иста "замрзни го завршениот агент на IDLE" логика како train.py -
        # спречува epsilon-greedy истражување врз возило кое веќе се судрило/стигнало
        active = np.ones(num_agents, dtype=bool)
        while not done_all:
            actions = agent.get_actions(states, epsilon)
            actions = [a if active[i] else IDLE_ACTION for i, a in enumerate(actions)]
            next_states, rewards, dones, info = env.step(actions)
            agent.update_memory(states, actions, rewards, next_states, dones)
            agent.train_step()
            active = active & ~np.array(dones)
            states = next_states
            done_all = all(dones)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        if (ep + 1) % target_update_every == 0:
            agent.update_target_model()

    # ---- евалуација ----
    # Забелешка за courtesy penalty при мерење: за ablation #1
    # (courtesy_weight=0.0 секогаш) наградата е "чиста" без shaping.
    # За ablation #2, courtesy penalty е ВКЛУЧЕНА и во тренингот И во
    # евалуацијата (бидејќи courtesy_weight веќе е поставен во самиот env
    # уште при конструкцијата) - ова е намерно, вака мерам колку реално
    # чини (во reward/arrival_rate) кога агентот навистина се однесува
    # "пофер", не само хипотетски.
    eval_rewards, collisions, arrivals = [], [], []
    for ep in range(eval_episodes):
        states = env.reset(seed=EVAL_SEED_OFFSET + seed + ep)
        ep_reward = np.zeros(num_agents)
        collided = False
        n_arrived = 0
        done_all = False
        # иста "замрзни го завршениот агент на IDLE" логика како во
        # тренинг-петлата погоре и train.py - претходно овој eval loop ја
        # немаше (инконзистентност train/eval); наградата веќе не се
        # "истекува" (envs/multi_agent_intersection.py поправка), но
        # акцијата сепак треба да е конзистентна.
        active = np.ones(num_agents, dtype=bool)
        while not done_all:
            actions = agent.get_actions(states, epsilon=0.0)  # чиста политика, без истражување
            actions = [a if active[i] else IDLE_ACTION for i, a in enumerate(actions)]
            states, rewards, dones, info = env.step(actions)
            ep_reward += np.array(rewards)
            collided = collided or any(info["crashed"])
            n_arrived = sum(info["arrived"])
            active = active & ~np.array(dones)
            done_all = all(dones)
        eval_rewards.append(float(ep_reward.mean()))
        collisions.append(collided)
        arrivals.append(n_arrived / num_agents)

    env.close()

    return {
        "mean_reward": float(np.mean(eval_rewards)),
        "std_reward": float(np.std(eval_rewards)),
        "collision_rate": float(np.mean(collisions)),
        "arrival_rate": float(np.mean(arrivals)),
    }


def _collision_job(job):
    # top-level функција (не внатрешна/lambda) - потребно за да може
    # multiprocessing да ја "pickle"-а при паралелно стартување
    w, seed, train_episodes, eval_episodes, num_agents = job
    r = train_and_eval_vdn(
        config_overrides={"collision_reward": w}, courtesy_weight=0.0, num_agents=num_agents,
        train_episodes=train_episodes, eval_episodes=eval_episodes, seed=seed,
    )
    return w, seed, r


def _courtesy_job(job):
    cw, seed, train_episodes, eval_episodes, num_agents = job
    r = train_and_eval_vdn(
        config_overrides={}, courtesy_weight=cw, num_agents=num_agents,
        train_episodes=train_episodes, eval_episodes=eval_episodes, seed=seed,
    )
    return cw, seed, r


def ablation_collision_weight(train_episodes, eval_episodes, seeds, num_agents, max_workers=None):
    print("\n########## ABLATION 1: тежина на collision_reward ##########")
    weights = [-5, -10, -15, -25]
    jobs = [(w, seed, train_episodes, eval_episodes, num_agents) for w in weights for seed in seeds]
    job_results = run_jobs(_collision_job, jobs, max_workers=max_workers, label="collision_reward×seed")

    results = {}
    for w in weights:
        runs = [r for (jw, jseed, r) in job_results if jw == w]
        agg = {k: agg_stats([r[k] for r in runs]) for k in runs[0]}
        results[str(w)] = {"per_seed": runs, "aggregate": agg}

    print("\n--- Резиме: collision_reward ablation ---")
    print(f"{'collision_reward':<18}{'reward':<20}{'collision_rate':<20}{'arrival_rate':<20}")
    for w in weights:
        agg = results[str(w)]["aggregate"]
        print(
            f"{w:<18}"
            f"{agg['mean_reward']['mean']:.2f} ± {agg['mean_reward']['ci95']:.2f}    "
            f"{agg['collision_rate']['mean']:.2f} ± {agg['collision_rate']['ci95']:.2f}      "
            f"{agg['arrival_rate']['mean']:.2f} ± {agg['arrival_rate']['ci95']:.2f}"
        )
    return results


def ablation_courtesy_weight(train_episodes, eval_episodes, seeds, num_agents, max_workers=None):
    print("\n########## ABLATION 2: courtesy (fairness) shaping член ##########")
    courtesy_weights = [0.0, 1.0, 3.0]
    jobs = [(cw, seed, train_episodes, eval_episodes, num_agents) for cw in courtesy_weights for seed in seeds]
    job_results = run_jobs(_courtesy_job, jobs, max_workers=max_workers, label="courtesy_weight×seed")

    results = {}
    for cw in courtesy_weights:
        runs = [r for (jcw, jseed, r) in job_results if jcw == cw]
        agg = {k: agg_stats([r[k] for r in runs]) for k in runs[0]}
        results[str(cw)] = {"per_seed": runs, "aggregate": agg}

    print("\n--- Резиме: courtesy weight ablation ---")
    print(f"{'courtesy_weight':<18}{'reward':<20}{'collision_rate':<20}{'arrival_rate':<20}")
    for cw in courtesy_weights:
        agg = results[str(cw)]["aggregate"]
        print(
            f"{cw:<18}"
            f"{agg['mean_reward']['mean']:.2f} ± {agg['mean_reward']['ci95']:.2f}    "
            f"{agg['collision_rate']['mean']:.2f} ± {agg['collision_rate']['ci95']:.2f}      "
            f"{agg['arrival_rate']['mean']:.2f} ± {agg['arrival_rate']['ci95']:.2f}"
        )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500, help="тренинг епизоди по конфигурација/seed")
    parser.add_argument("--eval_episodes", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--num_agents", type=int, default=3)
    parser.add_argument("--only", choices=["collision", "courtesy", "both"], default="both")
    parser.add_argument("--workers", type=int, default=None, help="колку паралелни процеси (default: сите јадра - 2)")
    args = parser.parse_args()

    out = {}
    if args.only in ("collision", "both"):
        out["collision_weight_ablation"] = ablation_collision_weight(
            args.episodes, args.eval_episodes, args.seeds, args.num_agents, args.workers
        )
    if args.only in ("courtesy", "both"):
        out["courtesy_weight_ablation"] = ablation_courtesy_weight(
            args.episodes, args.eval_episodes, args.seeds, args.num_agents, args.workers
        )

    out_path = RESULTS_DIR / "reward_ablation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nРезултатите од ablation студијата зачувани во: {out_path}")
