"""
Овде ги генерирам публикациски-квалитетните графици за проектниот извештај.

Прави 9 фигури, секоја е замислена да оди директно во извештајот:
  1. training_curves.png        - конвергенција на сите методи (reward/collision/arrival)
  2. method_comparison.png      - групиран bar chart со 95% доверителен интервал
  3. radar_comparison.png       - "spider" дијаграм - нормализирана мулти-метричка споредба
  4. reward_ablation.png        - однос безбедност-ефикасност при варирање на collision_reward
  5. courtesy_ablation.png      - цена на courtesy_weight (мојот fairness shaping член)
  6. scalability.png            - перформанси по број контролирани возила (2-5 агенти)
  7. hyperparameter_study.png   - one-at-a-time варирање на learning_rate/gamma/epsilon (VDN)
  8. obs_mode_comparison.png    - advanced студија: Kinematics/Pixels/Fusion × IQL/VDN
  9. robustness_study.png       - advanced студија: пад на перформанси под шум/blur/darken/branch-dropout

Принципи на кои се држам при секој график (научив дека овие работи се
важни за читливост во печатен извештај):
  - Иста боја за секој метод низ СИТЕ фигури - полесно се препознава без
    постојано да се гледа во легендата
  - Sans-serif фонтови, доволно големи (>=11pt) за да се читаат и на хартија
  - Секоја фигура има наслов, означени оски, легенда
  - Error bars / доверителни интервали секаде каде има повеќе seed-ови -
    без тоа, читателот не може да знае колку е "сериозна" разликата
  - Висока резолуција (300 DPI) - да не биде матно кога ќе се отпечати
  - Секоја бројка на графикон доаѓа од конкретен JSON фајл - лесно се
    проверува откаде дошол секој број, нема "магични" бројки
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch

# Кирилски print() на не-UTF8 конзола фрла UnicodeEncodeError - истата
# причина како во train.py, види коментар таму.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True, parents=True)

# за фигура 9 (robustness) - истиот "кои услови важат за кој obs_mode"
# речник како во experiments/robustness_study.py, за да не се дуплира тука.
from experiments.robustness import APPLICABLE_CONDITIONS

# ---------------------------------------------------------------------------
# Заеднички стил - го применувам на сите фигури во овој фајл, за конзистентност
# ---------------------------------------------------------------------------
COLORS = {
    "heuristic": "#7f8c8d",   # неутрално сива - "без учење" baseline
    "iql": "#e67e22",         # портокалова - независни агенти
    "vdn": "#2980b9",         # сина - CTDE (мојот главен придонес)
}
LABELS = {
    "heuristic": "Heuristic\n(rule-based)",
    "iql": "IQL\n(независни агенти)",
    "vdn": "VDN\n(CTDE)",
}
LABELS_SHORT = {
    "heuristic": "Heuristic",
    "iql": "IQL",
    "vdn": "VDN",
}
METHOD_ORDER = ["heuristic", "iql", "vdn"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "legend.fontsize": 9.5,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
})


def _moving_average(x, window=15):
    # movingaverage за да се "измазни" крива и полесно да се види трендот
    # низ шумот - секоја епизода поединечно скока многу (стохастична околина)
    x = np.asarray(x, dtype=float)
    if len(x) < window:
        return np.arange(len(x)), x
    ma = np.convolve(x, np.ones(window) / window, mode="valid")
    return np.arange(window - 1, len(x)), ma


def _load_json(path: Path):
    # ако фајлот сè уште не постои (пр. некој експеримент сè уште не е
    # пуштен), само враќам None и графикот се прескокнува - не сакам скриптата да падне
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _find_history_paths(method: str) -> list[Path]:
    """
    Ги наоѓа СИТЕ достапни тренинг-историски фајлови за даден метод, за
    training_curves.png фигурата - не само еден.

    Прво пробувам tag-ираните results/{method}_seed*_history.json (од
    паралелно run_comparison.py - обично 5 фајлови, еден по seed). Ако
    нема ниту еден таков, паѓам назад на "голото" results/{method}_history.json
    (директно `python train.py --method vdn`, без --tag).

    ПОПРАВКА: претходно земав само ЕДЕН (најнискиот) достапен seed, што
    даваше многу шумна крива - секоја епизода поединечно многу варира
    (стохастична околина), а movingaverage(15) врз само 1 seed не е
    доволно за јасно да се види трендот низ 2000 епизоди. Штом веќе
    имам 5 завршени seed-ови, просекот низ нив (пред moving average) дава
    многу почист сигнал (пад на шумот ~sqrt(n_seeds)), без да ја сменам
    методологијата - method_comparison.png со агрегирани ФИНАЛНИ статистики
    останува одделно, ова е само подобра ВИЗУЕЛИЗАЦИЈА на конвергенцијата.
    """
    tagged = sorted(RESULTS_DIR.glob(f"{method}_seed*_history.json"))
    if tagged:
        return tagged
    bare = RESULTS_DIR / f"{method}_history.json"
    return [bare] if bare.exists() else []


# ---------------------------------------------------------------------------
# 1. Тренинг криви - конвергенција низ сите методи, едно до друго
# ---------------------------------------------------------------------------
def plot_training_curves(methods=METHOD_ORDER, window=50):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    metrics = [
        ("mean_reward", "Просечна награда по агент", axes[0], False),
        ("collision_flag", "Стапка на судири", axes[1], True),
        ("arrival_flag", "Стапка на успешно пристигнување", axes[2], True),
    ]

    any_data = False
    for method in methods:
        hists = [h for p in _find_history_paths(method) if (h := _load_json(p))]
        if not hists:
            continue
        any_data = True

        # ако seed-овите имаат различен број епизоди (пр. нецелосен run),
        # ги сечам сите на најкраткиот - треба да порамнуваат по епизода
        # за да смее да се просечи низ нив
        n_eps = min(len(h) for h in hists)
        episodes = [hists[0][i]["episode"] for i in range(n_eps)]

        rewards_stack = np.array([[h[i]["mean_reward"] for i in range(n_eps)] for h in hists])
        collided_stack = np.array(
            [[1.0 if h[i]["collisions"] > 0 else 0.0 for i in range(n_eps)] for h in hists]
        )
        arrived_stack = np.array(
            [[1.0 if h[i]["arrivals"] > 0 else 0.0 for i in range(n_eps)] for h in hists]
        )
        # просек низ seed-овите, по епизода - ова е "суровиот" сигнал сега
        # (порано беше суров сигнал од само 1 seed)
        data_map = {
            "mean_reward": rewards_stack.mean(axis=0),
            "collision_flag": collided_stack.mean(axis=0),
            "arrival_flag": arrived_stack.mean(axis=0),
        }
        n_seeds = len(hists)
        label = LABELS[method].replace("\n", " ")
        if n_seeds > 1:
            label += f" ({n_seeds} seed-а)"

        for key, _, ax, _ in metrics:
            idx, ma = _moving_average(data_map[key], window)
            raw = data_map[key]
            # тенка/проѕирна линија за просекот-низ-seed-ови по епизода, задебелена за movingaverage-от
            ax.plot(episodes, raw, color=COLORS[method], alpha=0.2, linewidth=0.8)
            ax.plot(
                [episodes[i] for i in idx], ma,
                color=COLORS[method], linewidth=2.2, label=label,
            )

    if not any_data:
        plt.close(fig)
        print("[training_curves] Нема достапни *_history.json фајлови - прескокнато.")
        return None

    for key, title, ax, is_rate in metrics:
        ax.set_title(title)
        ax.set_xlabel("Епизода")
        if is_rate:
            ax.set_ylim(-0.03, 1.03)
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Награда")
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper center", ncol=len(methods), bbox_to_anchor=(0.5, 1.08))
    fig.suptitle(
        f"Конвергенција на тренингот (movingaverage, прозорец={window} епизоди)",
        y=1.16, fontsize=13, fontweight="bold",
    )
    out = FIGURES_DIR / "training_curves.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[OK] {out}")
    return out


# ---------------------------------------------------------------------------
# 2. Финална споредба - групиран bar chart со 95% доверителен интервал
# ---------------------------------------------------------------------------
def plot_method_comparison():
    summary = _load_json(RESULTS_DIR / "comparison_summary.json")
    if not summary:
        print("[method_comparison] Нема comparison_summary.json - прескокнато.")
        return None

    agg = summary["aggregate"]
    methods = [m for m in METHOD_ORDER if m in agg]

    panels = [
        ("reward", "Просечна награда", False),
        ("collision_rate", "Стапка на судири", True),
        ("arrival_rate", "Стапка на успех", True),
        ("avg_steps", "Просечни чекори", False),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.5))
    for (key, title, is_rate), ax in zip(panels, axes):
        means, cis = [], []
        for m in methods:
            stats = agg[m][key]  # {"mean": ..., "ci95": ...}
            means.append(stats["mean"])
            cis.append(stats["ci95"])

        x = np.arange(len(methods))
        bars = ax.bar(
            x, means, yerr=cis, capsize=5,
            color=[COLORS[m] for m in methods],
            edgecolor="white", linewidth=1.2,
            error_kw={"elinewidth": 1.3, "ecolor": "#333333"},
        )
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS_SHORT[m] for m in methods], fontsize=10.5)
        ax.set_title(title)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if is_rate:
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        for rect, mean in zip(bars, means):
            label = f"{mean:.0%}" if is_rate else f"{mean:.2f}"
            ax.annotate(
                label, xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9.5, fontweight="bold",
            )

    handles = [Patch(color=COLORS[m], label=LABELS[m].replace("\n", " ")) for m in methods]
    fig.legend(handles=handles, loc="upper center", ncol=len(methods), bbox_to_anchor=(0.5, 1.1), fontsize=10.5)
    fig.suptitle(
        "Финална споредба на методите (средина ± 95% CI низ seed-ови)",
        y=1.2, fontsize=13, fontweight="bold",
    )
    out = FIGURES_DIR / "method_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[OK] {out}")
    return out


# ---------------------------------------------------------------------------
# 3. Radar / spider дијаграм - нормализирана мулти-метричка споредба
#    (визуелно веднаш се гледа целосниот "профил" на секој метод)
# ---------------------------------------------------------------------------
def plot_radar_comparison():
    summary = _load_json(RESULTS_DIR / "comparison_summary.json")
    if not summary:
        print("[radar_comparison] Нема comparison_summary.json - прескокнато.")
        return None

    agg = summary["aggregate"]
    methods = [m for m in METHOD_ORDER if m in agg]

    # метрики каде "повисоко = подобро" за конзистентен radar (безбедност = 1 - стапка на судири)
    def get_mean(m, key):
        return float(agg[m][key]["mean"])

    categories = ["Награда", "Безбедност\n(1 - судири)", "Стапка на\nуспех", "Ефикасност\n(инверзно чекори)"]
    raw = {m: [] for m in methods}
    for m in methods:
        reward = get_mean(m, "reward")
        safety = 1.0 - get_mean(m, "collision_rate")
        arrival = get_mean(m, "arrival_rate")
        steps = get_mean(m, "avg_steps")
        efficiency = 1.0 / steps if steps > 0 else 0.0
        raw[m] = [reward, safety, arrival, efficiency]

    # нормализација по метрика (min-max низ методите), стеснета во опсег
    # [0.25, 1] наместо [0.05, 1] - за графикот да не изгледа "драматично
    # исцртан до раб" кога разликите меѓу методите се реално мали
    raw_arr = np.array([raw[m] for m in methods])
    mins = raw_arr.min(axis=0)
    maxs = raw_arr.max(axis=0)
    ranges = np.where(maxs - mins < 1e-9, 1.0, maxs - mins)
    norm = (raw_arr - mins) / ranges * 0.65 + 0.25

    n = len(categories)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True))
    for i, m in enumerate(methods):
        values = norm[i].tolist()
        values += values[:1]
        ax.plot(angles, values, color=COLORS[m], linewidth=2.2, label=LABELS[m].replace("\n", " "))
        ax.fill(angles, values, color=COLORS[m], alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10.5)
    ax.set_yticklabels([])
    ax.set_ylim(0, 1)
    ax.spines["polar"].set_alpha(0.3)
    ax.grid(alpha=0.3)
    ax.set_title(
        "Мулти-метрички профил (нормализирано низ методите)\n"
        "поголема површина = подобар сеопфатен резултат",
        fontsize=12, fontweight="bold", pad=25,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))
    out = FIGURES_DIR / "radar_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[OK] {out}")
    return out


# ---------------------------------------------------------------------------
# 4. Reward ablation - однос безбедност/ефикасност при варирање на collision_reward
# ---------------------------------------------------------------------------
def plot_reward_ablation():
    data = _load_json(RESULTS_DIR / "reward_ablation.json")
    if not data or "collision_weight_ablation" not in data:
        print("[reward_ablation] Нема reward_ablation.json - прескокнато.")
        return None

    cw_data = data["collision_weight_ablation"]
    weights = sorted([float(k) for k in cw_data.keys()])

    coll_means, coll_cis = [], []
    arr_means, arr_cis = [], []
    rew_means, rew_cis = [], []
    for w in weights:
        agg = cw_data[str(int(w))]["aggregate"] if str(int(w)) in cw_data else cw_data[str(w)]["aggregate"]
        # agg_stats() (experiments/parallel.py) враќа {"mean":..,"ci95":..}
        # dict, НЕ (mean, ci) tuple - reward_ablation.json сега го користи
        # истиот формат како comparison_summary.json/scalability_study.json
        coll_means.append(agg["collision_rate"]["mean"]); coll_cis.append(agg["collision_rate"]["ci95"])
        arr_means.append(agg["arrival_rate"]["mean"]); arr_cis.append(agg["arrival_rate"]["ci95"])
        rew_means.append(agg["mean_reward"]["mean"]); rew_cis.append(agg["mean_reward"]["ci95"])

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()  # втора y-оска, за да можам да ги прикажам двете метрики на ист график

    l1 = ax1.errorbar(
        weights, coll_means, yerr=coll_cis, marker="o", markersize=7,
        color="#c0392b", linewidth=2.2, capsize=4, label="Стапка на судири (безбедност)",
    )
    l2 = ax2.errorbar(
        weights, arr_means, yerr=arr_cis, marker="s", markersize=7,
        color="#27ae60", linewidth=2.2, capsize=4, linestyle="--", label="Стапка на успех (ефикасност)",
    )

    ax1.set_xlabel("collision_reward (тежина на казна за судир)")
    ax1.set_ylabel("Стапка на судири", color="#c0392b")
    ax2.set_ylabel("Стапка на успешно пристигнување", color="#27ae60")
    ax1.tick_params(axis="y", labelcolor="#c0392b")
    ax2.tick_params(axis="y", labelcolor="#27ae60")
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    lines = [l1, l2]
    labels_ = [line.get_label() for line in lines]
    ax1.legend(lines, labels_, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    ax1.set_title(
        "Ablation: однос безбедност-ефикасност при различна тежина на collision_reward\n(VDN агент)",
        fontsize=12.5, fontweight="bold",
    )
    out = FIGURES_DIR / "reward_ablation.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[OK] {out}")
    return out


# ---------------------------------------------------------------------------
# 5. Courtesy ablation
# ---------------------------------------------------------------------------
def plot_courtesy_ablation():
    data = _load_json(RESULTS_DIR / "reward_ablation.json")
    if not data or "courtesy_weight_ablation" not in data:
        print("[courtesy_ablation] Нема courtesy ablation податоци - прескокнато.")
        return None

    cw_data = data["courtesy_weight_ablation"]
    weights = sorted([float(k) for k in cw_data.keys()])

    def agg_for(w):
        key = str(w) if str(w) in cw_data else str(int(w)) if str(int(w)) in cw_data else None
        return cw_data[key]["aggregate"]

    # ПОПРАВКА: претходно оваа фигура прикажуваше само reward+arrival_rate,
    # иако collision_rate ВЕЌЕ постои во reward_ablation.json (се
    # пресметува во ablation_courtesy_weight()) - едноставно не се
    # користеше тука. Дадено дека целата поента на courtesy shaping е
    # безбедност/фер-ност кон human-driven возилата, отсуство на
    # collision_rate токму од ОВОЈ график беше пропуст. Претворено во 3
    # панели (исто како method_comparison/scalability) наместо twin-axis
    # bar chart, за да стане чисто и конзистентно со останатите фигури.
    panels = [
        ("mean_reward", "Просечна награда", False),
        ("collision_rate", "Стапка на судири", True),
        ("arrival_rate", "Стапка на успех", True),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6))
    x = np.arange(len(weights))
    for (key, title, is_rate), ax in zip(panels, axes):
        means = [agg_for(w)[key]["mean"] for w in weights]
        cis = [agg_for(w)[key]["ci95"] for w in weights]
        ax.bar(x, means, yerr=cis, capsize=4, color="#16a085", edgecolor="white", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels([f"w={w:g}" for w in weights])
        ax.set_title(title)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if is_rate:
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    fig.supxlabel("courtesy_weight (тежина на fairness shaping членот)", fontsize=10.5)
    fig.suptitle(
        "Ablation: цена на 'фер' однесување кон human-driven возила\n(VDN агент)",
        y=1.05, fontsize=12.5, fontweight="bold",
    )
    out = FIGURES_DIR / "courtesy_ablation.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[OK] {out}")
    return out


# ---------------------------------------------------------------------------
# 6. Скалабилност - перформанси по број контролирани возила (агенти)
# ---------------------------------------------------------------------------
def plot_scalability():
    data = _load_json(RESULTS_DIR / "scalability_study.json")
    if not data:
        print("[scalability] Нема scalability_study.json - прескокнато.")
        return None

    agent_counts = data["agent_counts"]
    agg = data["aggregate"]
    methods = [m for m in METHOD_ORDER if m in agg]

    panels = [
        ("reward", "Просечна награда", False),
        ("collision_rate", "Стапка на судири", True),
        ("arrival_rate", "Стапка на успех", True),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for (key, title, is_rate), ax in zip(panels, axes):
        for m in methods:
            means = [agg[m][str(n)][key]["mean"] for n in agent_counts]
            cis = [agg[m][str(n)][key]["ci95"] for n in agent_counts]
            ax.errorbar(agent_counts, means, yerr=cis, marker="o", markersize=6,
                        linewidth=2.2, color=COLORS[m], capsize=4, label=LABELS_SHORT[m])
        ax.set_title(title)
        ax.set_xlabel("Број контролирани возила")
        ax.set_xticks(agent_counts)
        if is_rate:
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles = [Patch(color=COLORS[m], label=LABELS_SHORT[m]) for m in methods]
    fig.legend(handles=handles, loc="upper center", ncol=len(methods), bbox_to_anchor=(0.5, 1.08))
    fig.suptitle(
        "Скалабилност: перформанси по број контролирани возила (агенти)",
        y=1.16, fontsize=13, fontweight="bold",
    )
    out = FIGURES_DIR / "scalability.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[OK] {out}")
    return out


# ---------------------------------------------------------------------------
# 7. Хиперпараметарска студија (VDN) - one-at-a-time варирање од baseline
# ---------------------------------------------------------------------------
def plot_hyperparameter_study():
    data = _load_json(RESULTS_DIR / "hyperparameter_study.json")
    if not data:
        print("[hyperparameter_study] Нема hyperparameter_study.json - прескокнато.")
        return None

    agg = data["aggregate"]
    slugs = list(agg.keys())  # Python dict-от го чува редоследот на вметнување - истиот редослед како CONFIGS
    labels = [agg[s]["label"] for s in slugs]
    x = np.arange(len(slugs))
    # baseline конфигурацијата истакната со друга боја за лесно да се препознае на графикот
    bar_colors = [COLORS["vdn"] if s == "baseline" else "#95a5a6" for s in slugs]

    panels = [
        ("reward", "Просечна награда", False),
        ("collision_rate", "Стапка на судири", True),
        ("arrival_rate", "Стапка на успех", True),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    for (key, title, is_rate), ax in zip(panels, axes):
        means = [agg[s][key]["mean"] for s in slugs]
        cis = [agg[s][key]["ci95"] for s in slugs]
        ax.bar(x, means, yerr=cis, capsize=4, color=bar_colors, edgecolor="white", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8.5)
        ax.set_title(title)
        if is_rate:
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Хиперпараметарска студија (VDN) - one-at-a-time варирање од baseline",
        y=1.06, fontsize=13, fontweight="bold",
    )
    out = FIGURES_DIR / "hyperparameter_study.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[OK] {out}")
    return out


# ---------------------------------------------------------------------------
# 8. Observation-mode споредба - advanced дополнителна студија
#    (Kinematics/Pixels/Fusion × IQL/VDN, experiments/observation_study.py)
# ---------------------------------------------------------------------------
OBS_MODE_ORDER = ["kinematics", "pixels", "fusion"]
OBS_MODE_LABELS = {"kinematics": "Kinematics", "pixels": "Pixels", "fusion": "Fusion"}


def plot_obs_mode_comparison():
    data = _load_json(RESULTS_DIR / "observation_comparison.json")
    if not data:
        print("[obs_mode_comparison] Нема observation_comparison.json - прескокнато "
              "(прво пушти experiments/observation_study.py).")
        return None

    agg = data["aggregate"]
    obs_modes = [om for om in OBS_MODE_ORDER if om in agg]
    methods = [m for m in ("iql", "vdn") if any(m in agg.get(om, {}) for om in obs_modes)]

    panels = [
        ("reward", "Просечна награда", False),
        ("collision_rate", "Стапка на судири", True),
        ("arrival_rate", "Стапка на успех", True),
        ("avg_steps", "Просечни чекори", False),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.6))
    bar_width = 0.35
    x = np.arange(len(obs_modes))
    for (key, title, is_rate), ax in zip(panels, axes):
        for j, m in enumerate(methods):
            means, cis = [], []
            for om in obs_modes:
                row = agg.get(om, {}).get(m)
                means.append(row[key]["mean"] if row else 0.0)
                cis.append(row[key]["ci95"] if row else 0.0)
            offset = (j - (len(methods) - 1) / 2) * bar_width
            ax.bar(x + offset, means, bar_width, yerr=cis, capsize=4,
                   color=COLORS[m], edgecolor="white", linewidth=1.0, label=LABELS_SHORT[m])
        ax.set_xticks(x)
        ax.set_xticklabels([OBS_MODE_LABELS[om] for om in obs_modes])
        ax.set_title(title)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if is_rate:
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    handles = [Patch(color=COLORS[m], label=LABELS_SHORT[m]) for m in methods]
    fig.legend(handles=handles, loc="upper center", ncol=len(methods), bbox_to_anchor=(0.5, 1.08))
    fig.suptitle(
        "Advanced студија: Kinematics vs Pixels vs Fusion опсервации (IQL/VDN, средина ± 95% CI)",
        y=1.16, fontsize=13, fontweight="bold",
    )
    out = FIGURES_DIR / "obs_mode_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[OK] {out}")
    return out


# ---------------------------------------------------------------------------
# 9. Robustness студија - пад (%) на перформанси под шум/blur/darken/
#    branch-dropout, наспроти clean евалуација (experiments/robustness_study.py)
# ---------------------------------------------------------------------------
CONDITION_LABELS = {
    "kin_noise": "kin шум",
    "pixel_blur": "blur",
    "pixel_dark": "darken",
    "drop_kin": "drop kin",
    "drop_img": "drop img",
}


def plot_robustness_study():
    data = _load_json(RESULTS_DIR / "robustness_study.json")
    if not data:
        print("[robustness_study] Нема robustness_study.json - прескокнато "
              "(прво пушти experiments/robustness_study.py).")
        return None

    agg = data["aggregate"]
    methods = ["iql", "vdn"]
    obs_modes = [om for om in OBS_MODE_ORDER if any(f"{m}/{om}" in agg for m in methods)]
    if not obs_modes:
        print("[robustness_study] Нема употребливи податоци во robustness_study.json - прескокнато.")
        return None

    # позитивно = ПОЛОШО од clean (сметано веќе во robustness_study.py::DIRECTION)
    metrics = [("mean_reward", "Пад на награда (%)"), ("collision_rate", "Раст на судири (%)")]
    fig, axes = plt.subplots(len(metrics), len(obs_modes),
                              figsize=(4.6 * len(obs_modes), 4 * len(metrics)), squeeze=False)

    bar_width = 0.35
    for row, (metric_key, row_title) in enumerate(metrics):
        for col, om in enumerate(obs_modes):
            ax = axes[row][col]
            conditions = APPLICABLE_CONDITIONS[om]
            x = np.arange(len(conditions))
            for j, m in enumerate(methods):
                by_cond = agg.get(f"{m}/{om}", {})
                means, cis = [], []
                for cond in conditions:
                    entry = by_cond.get(cond)
                    stats = entry["pct_change_vs_clean"][metric_key] if entry else {"mean": 0.0, "ci95": 0.0}
                    means.append(stats["mean"])
                    cis.append(stats["ci95"])
                offset = (j - (len(methods) - 1) / 2) * bar_width
                ax.bar(x + offset, means, bar_width, yerr=cis, capsize=3,
                       color=COLORS[m], edgecolor="white", linewidth=0.8, label=LABELS_SHORT[m])
            ax.axhline(0, color="#333333", linewidth=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels([CONDITION_LABELS.get(c, c) for c in conditions], fontsize=9, rotation=20, ha="right")
            if col == 0:
                ax.set_ylabel(row_title)
            if row == 0:
                ax.set_title(OBS_MODE_LABELS[om])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    handles = [Patch(color=COLORS[m], label=LABELS_SHORT[m]) for m in methods]
    fig.legend(handles=handles, loc="upper center", ncol=len(methods), bbox_to_anchor=(0.5, 1.04))
    fig.suptitle(
        "Robustness студија: пад на перформанси наспроти clean евалуација (позитивно = полошо)",
        y=1.08, fontsize=13, fontweight="bold",
    )
    out = FIGURES_DIR / "robustness_study.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[OK] {out}")
    return out


if __name__ == "__main__":
    print("Генерирам публикациски-квалитетни фигури во results/figures/ ...\n")
    plot_training_curves()
    plot_method_comparison()
    plot_radar_comparison()
    plot_reward_ablation()
    plot_courtesy_ablation()
    plot_scalability()
    plot_hyperparameter_study()
    plot_obs_mode_comparison()
    plot_robustness_study()
    print("\nГотово.")
