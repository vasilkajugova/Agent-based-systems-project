"""
Главен тренинг скрипт - оттука ги тренирам сите методи.

Употреба:
    python train.py --method iql   --episodes 200
    python train.py --method vdn   --episodes 200
    python train.py --method heuristic --episodes 20   # само евалуација, нема тренинг (heuristic не учи)

Намерно е конфигурирано за БРЗ тренинг/тест циклус, за да можам често да
проверувам дали сè уште работи сè додека го развивам проектот:
  - duration=20s по епизода (кратка сцена на раскрсница)
  - policy_frequency стандардно highway-env ~ неколку чекори во секунда
  -> целосна епизода трае типично <100 чекори, стотици епизоди завршуваат
     за само неколку минути на обичен CPU (без GPU).

За поголеми експерименти (повеќе seed-ови/конфигурации истовремено), види
experiments/parallel.py - го користат experiments/run_comparison.py,
reward_ablation.py, scalability_study.py и hyperparameter_study.py за да
ги тренираат независните running-ови паралелно на повеќе CPU јадра
наместо еден по еден.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from envs.multi_agent_intersection import MultiAgentIntersectionEnv
from agents.iql_manager import IQLManager
from agents.vdn_agent import VDNAgent
from agents.heuristic_agent import HeuristicAgent

# Кирилски print() на не-UTF8 Windows конзола (пр. cp1252) фрла
# UnicodeEncodeError и го урива целиот процес среде долг тренинг - истата
# класа грешка како "грчките букви" случајот подолу во
# experiments/hyperparameter_study.py, само тука со самата кирилица, не
# само со грчки букви. errors="replace" значи ако сепак некој нов, пореток
# знак не се впише, го заменува со "?" наместо да падне целиот процес.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass  # sys.stdout можеби не поддржува reconfigure (пр. некои IDE конзоли) - ОК, продолжуваме

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Индекс на IDLE акцијата во DiscreteMetaAction (envs/multi_agent_intersection.py
# DEFAULT_CONFIG, исто како agents/heuristic_agent.py::ACTIONS). Го користам
# за да "замрзнам" веќе завршени агенти на IDLE наместо да продолжат со
# случајни акции - види коментар подолу во run_training().
IDLE_ACTION = 1

# Овде дефинирам "безбедна" разлика помеѓу seed-овите за тренинг
# (env.reset(seed=seed+ep) подолу, за ep во опсег [0, episodes)) и
# seed-овите за евалуација, користени во experiments/run_comparison.py и
# experiments/reward_ablation.py.
#
# Зошто ми е ова важно: ако немам голема, фиксна разлика помеѓу нив, а
# бројот на тренинг епизоди некогаш го надмине EVAL_SEED_OFFSET, постои
# ризик евалуацијата да користи ИСТ env.reset(seed=...) како некоја
# тренинг епизода - тогаш агентот всушност би бил "тестиран" на сценарио
# кое веќе го видел при тренирање (data leakage), што би ги направило
# резултатите невалидни. 1 000 000 е далеку над и најдолгиот тренинг што
# го планирам (README спомнува до 3000 епизоди), па сум сигурна дека
# нема да се преклопат.
EVAL_SEED_OFFSET = 1_000_000


def set_global_seed(seed: int):
    """
    Seed-ирам ги СИТЕ извори на случајност - и numpy И torch. Без ова,
    два стартувања со ист --seed сепак даваат различни резултати, бидејќи
    иницијализацијата на тежините на невронската мрежа (torch) си останува
    случајна ако не се seed-ира посебно. Ова ми беше една од првите грешки
    што ја најдов при проверка на репродуцибилноста (види README).
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_epsilon_decay(episodes: int, epsilon_start: float = 1.0, epsilon_min: float = 0.05,
                           target_fraction: float = 0.7) -> float:
    """
    Ја пресметува стапката на опаѓање на epsilon по епизода, така што
    epsilon стигнува до epsilon_min некаде околу `target_fraction` од
    целиот тренинг - СКАЛИРАНО автоматски со бројот на епизоди.

    Оваа функција ја користам и во train.py и во
    experiments/reward_ablation.py, за да имам иста шема на истражување
    (exploration) низ сите експерименти. Без ова, ако секоја скрипта си
    имаше свој hardcoded decay број, резултатите од различните
    експерименти воопшто не би биле методолошки споредливи - агент
    тргниран на 200 епизоди со decay пресметан за 2000 епизоди воопшто не
    би стигнал да "истражи" доволно.
    """
    target_episode = max(1, int(episodes * target_fraction))
    return (epsilon_min / epsilon_start) ** (1.0 / target_episode)


def make_manager(method: str, num_agents: int, obs_dim: int, num_actions: int, agent_kwargs: dict | None = None):
    # Едноставна "фабрика" - врз основа на текстуалното име на методот,
    # враќа соодветен објект кој знае да избира акции и (освен heuristic) да учи.
    # agent_kwargs (learning_rate, discount_factor, hidden, ...) ги
    # проследувам натаму до конструкторот на агентот - ова ми овозможи да
    # напишам experiments/hyperparameter_study.py без да морам да ја
    # копирам целата run_training() логика одделно за секој хиперпараметар.
    agent_kwargs = agent_kwargs or {}
    if method == "iql":
        return IQLManager(num_agents, obs_dim, num_actions, double_dqn=True, dueling=False, **agent_kwargs)
    elif method == "iql_dueling":
        # Достапна опција за рачно тестирање (`python train.py --method
        # iql_dueling ...`), НАМЕРНО не е вклучена во METHODS листите во
        # ниту еден experiments/*.py - секој дополнителен метод во главната
        # споредба ги удвојува (метод × seed) job-овите, а dueling
        # архитектурата веќе е тестирана преку VDN (dueling=True таму).
        # Ја задржувам тука само за флексибилност, не е "заборавен" код.
        return IQLManager(num_agents, obs_dim, num_actions, double_dqn=True, dueling=True, **agent_kwargs)
    elif method == "vdn":
        return VDNAgent(num_agents, obs_dim, num_actions, dueling=True, **agent_kwargs)
    elif method == "heuristic":
        return HeuristicAgent()
    else:
        raise ValueError(f"Непознат метод: {method}")


def run_training(
    method: str,
    num_agents: int = 3,
    episodes: int = 200,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: float | None = None,
    epsilon_target_fraction: float = 0.7,
    target_update_every: int = 10,
    seed: int = 0,
    verbose_every: int = 10,
    checkpoint_every: int = 100,
    tag: str | None = None,
    agent_kwargs: dict | None = None,
):
    set_global_seed(seed)
    env = MultiAgentIntersectionEnv(num_agents=num_agents)
    manager = make_manager(method, num_agents, env.obs_dim, env.num_actions, agent_kwargs=agent_kwargs)

    # Сите фајлови (checkpoint/history/model) ги именувам преку "name",
    # не директно преку "method". Ова е важно откако додадов паралелно
    # стартување (experiments/parallel.py): ако два процеси паралелно
    # тренираат ИСТ метод со РАЗЛИЧНИ seed-ови, а двата пишуваат во исто
    # results/{method}_history.json/_model, тоа е race condition - на
    # Windows дури фрла PermissionError бидејќи фајлот е заклучен додека
    # другиот процес пишува во него. Со tag (пр. "seed0", "seed1"), секој
    # паралелен job си добива свои, единствени патеки. tag=None (default)
    # го задржува старото едноставно однесување за директно стартување
    # (python train.py --method vdn ...) - тогаш name == method, точно
    # како порано.
    name = method if tag is None else f"{method}_{tag}"

    # epsilon_decay се пресметува автоматски врз основа на бројот епизоди
    # (compute_epsilon_decay погоре), освен ако не го зададам рачно - така
    # epsilon секогаш стигнува до минимумот околу epsilon_target_fraction
    # од тренингот, без разлика дали тренирам 150 или 2000 епизоди.
    if epsilon_decay is None:
        epsilon_decay = compute_epsilon_decay(episodes, epsilon_start, epsilon_min, epsilon_target_fraction)

    epsilon = epsilon_start
    history = []
    t0 = time.time()

    for ep in range(episodes):
        states = env.reset(seed=seed + ep)
        ep_rewards = np.zeros(num_agents)
        ep_collisions = 0
        ep_arrivals = 0
        ep_losses = []
        steps = 0
        done_all = False
        # Тимската епизода трае додека НЕ завршат СИТЕ агенти (done_all),
        # но секој поединечен агент може да заврши (се судри или стигне)
        # и порано од останатите. "active" го следи тоа - штом еден агент
        # заврши, неговата акција ја "замрзнувам" на IDLE наместо да
        # продолжи со epsilon-greedy истражување врз состојба која веќе
        # воопшто не влијае на исходот (нема смисла да трошам exploration
        # "буџет" на веќе судрено/пристигнато возило).
        active = np.ones(num_agents, dtype=bool)

        while not done_all:
            if method == "heuristic":
                actions = manager.get_actions(states)
            else:
                actions = manager.get_actions(states, epsilon)
            actions = [a if active[i] else IDLE_ACTION for i, a in enumerate(actions)]

            next_states, rewards, dones, info = env.step(actions)

            if method != "heuristic":
                manager.update_memory(states, actions, rewards, next_states, dones)
                loss = manager.train_step()
                # loss може да е float (VDN), листа (IQL - по еден loss за секој агент), или None
                if loss is not None:
                    if isinstance(loss, list):
                        losses = [l for l in loss if l is not None]
                        if losses:
                            ep_losses.append(float(np.mean(losses)))
                    else:
                        ep_losses.append(float(loss))

            # Тука бројам само НОВИ судири/пристигнувања - значи агент кој
            # ТОКМУ во ОВОЈ чекор премина од активен во done. Во првата
            # верзија на кодот погрешно го собирав sum(info["crashed"]) на
            # СЕКОЈ чекор откако возилото веќе се урнало (highway-env
            # внатрешно го чува crashed=True трајно до крајот на
            # епизодата), што ги "надуваше" суровите collisions/arrivals
            # броеви во history (не влијаеше на графиците, кои ги користат
            # само како бинарен сигнал >0, но сепак ги правеше самите
            # сурови бројки бесмислени во results/*_history.json).
            dones_arr = np.array(dones)
            newly_done = active & dones_arr
            ep_collisions += int((np.array(info["crashed"]) & newly_done).sum())
            ep_arrivals += int((np.array(info["arrived"]) & newly_done).sum())
            active = active & ~dones_arr

            ep_rewards += np.array(rewards)
            states = next_states
            steps += 1
            done_all = all(dones)

        if method != "heuristic":
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            if (ep + 1) % target_update_every == 0:
                manager.update_target_model()

        history.append(
            {
                "episode": ep,
                "mean_reward": float(ep_rewards.mean()),
                "collisions": int(ep_collisions),
                "arrivals": int(ep_arrivals),
                "steps": steps,
                "epsilon": round(epsilon, 3) if method != "heuristic" else None,
                "mean_loss": float(np.mean(ep_losses)) if ep_losses else None,
            }
        )

        if (ep + 1) % verbose_every == 0:
            recent = history[-verbose_every:]
            avg_r = np.mean([h["mean_reward"] for h in recent])
            coll_rate = np.mean([h["collisions"] > 0 for h in recent])
            recent_losses = [h["mean_loss"] for h in recent if h["mean_loss"] is not None]
            loss_str = f"  loss={np.mean(recent_losses):.3f}" if recent_losses else ""
            print(
                f"[{name}] еп {ep + 1}/{episodes}  "
                f"просечна награда={avg_r:.2f}  стапка на судири={coll_rate:.2f}  "
                f"eps={epsilon:.2f}{loss_str}  ({time.time() - t0:.1f}s поминато)"
            )

        # checkpoint - ги зачувувам моделот и историјата на секои
        # checkpoint_every епизоди, за да не изгубам сè ако тренингот падне
        # или го прекинам среде пат при подолги тренинзи. Секој нов
        # checkpoint ГО ПРЕЗАПИШУВА претходниот (иста патека, без број на
        # епизода во името) - намерно, за да не се насобираат десетици GB
        # непотребни фајлови при долги multi-seed/ablation тренинзи.
        if method != "heuristic" and (ep + 1) % checkpoint_every == 0:
            ckpt_prefix = RESULTS_DIR / f"{name}_checkpoint"
            manager.save(str(ckpt_prefix))
            with open(RESULTS_DIR / f"{name}_history_partial.json", "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)

    env.close()

    out_path = RESULTS_DIR / f"{name}_history.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"[{name}] Тренингот заврши за {time.time() - t0:.1f}s. Резултати: {out_path}")

    if method != "heuristic":
        model_path = RESULTS_DIR / f"{name}_model"
        manager.save(str(model_path))
        print(f"[{name}] Модел зачуван: {model_path}_*")

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["iql", "iql_dueling", "vdn", "heuristic"], required=True)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--num_agents", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", type=str, default=None, help="опционален суфикс за резултатски фајлови (за паралелни/повеќе running-ови)")
    args = parser.parse_args()

    run_training(
        method=args.method,
        num_agents=args.num_agents,
        episodes=args.episodes,
        seed=args.seed,
        tag=args.tag,
    )
