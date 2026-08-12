"""
Овој фајл ми служи за брза визуелна проверка на конвергенцијата/
стабилноста на еден тренинг (reward, стапка на судири, loss) со
movingaverage - го користам ПРЕД да "посветам" часови во голем тренинг со
многу епизоди/seed-ови, за да видам веднаш дали воопшто учи нешто пред да
чекам предолго.

Употреба:
    python plot_training_curves.py --method vdn
    python plot_training_curves.py --method iql --history_file results/iql_history_partial.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Кирилски print() на не-UTF8 конзола фрла UnicodeEncodeError - истата
# причина како во train.py, види коментар таму.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

RESULTS_DIR = Path(__file__).parent / "results"


def moving_average(x, window=20):
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")


def plot(method: str, history_file: str | None = None):
    path = Path(history_file) if history_file else RESULTS_DIR / f"{method}_history.json"
    with open(path, encoding="utf-8") as f:
        history = json.load(f)

    episodes = [h["episode"] for h in history]
    rewards = [h["mean_reward"] for h in history]
    collided = [1.0 if h["collisions"] > 0 else 0.0 for h in history]
    losses = [h["mean_loss"] for h in history if h.get("mean_loss") is not None]
    loss_episodes = [h["episode"] for h in history if h.get("mean_loss") is not None]

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=False)

    # --- Reward ---
    ax = axes[0]
    ax.plot(episodes, rewards, alpha=0.3, color="tab:blue", label="reward по епизода")
    if len(rewards) >= 20:
        ma = moving_average(rewards, 20)
        ax.plot(episodes[19:], ma, color="tab:blue", linewidth=2, label="movingaverage (20 еп.)")
    ax.set_ylabel("Просечна награда по агент")
    ax.set_title(f"[{method}] Тренинг крива на наградата")
    ax.legend()
    ax.grid(alpha=0.3)

    # --- Collision rate ---
    ax = axes[1]
    if len(collided) >= 20:
        ma = moving_average(collided, 20)
        ax.plot(episodes[19:], ma, color="tab:red", linewidth=2)
    ax.set_ylabel("Стапка на судири\n(movingaverage 20 еп.)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)

    # --- Loss ---
    ax = axes[2]
    if losses:
        ax.plot(loss_episodes, losses, alpha=0.3, color="tab:green")
        if len(losses) >= 20:
            ma = moving_average(losses, 20)
            ax.plot(loss_episodes[19:], ma, color="tab:green", linewidth=2)
        ax.set_yscale("log")  # log скала - loss-от опаѓа брзо на почеток, вака подобро се гледа целиот опсег
    ax.set_ylabel("TD loss (log скала)")
    ax.set_xlabel("Епизода")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = RESULTS_DIR / f"{method}_training_curves.png"
    plt.savefig(out_path, dpi=120)
    print(f"Графикон зачуван: {out_path}")

    # --- брза текстуална дијагноза - за да не морам секој пат да гледам ---
    # --- во самиот график, добивам веднаш текстуален заклучок ---
    print("\n--- Брза дијагноза ---")
    first_half = rewards[: len(rewards) // 2]
    second_half = rewards[len(rewards) // 2 :]
    print(f"Просечна награда прва половина: {np.mean(first_half):.2f}")
    print(f"Просечна награда втора половина: {np.mean(second_half):.2f}")
    if np.mean(second_half) > np.mean(first_half):
        print("=> Позитивен тренд - агентот учи.")
    else:
        print("=> ПРЕДУПРЕДУВАЊЕ: нема јасен раст на наградата. Провери хиперпараметри пред долг тренинг.")

    if losses and (np.isnan(losses[-1]) or losses[-1] > 10 * np.median(losses)):
        print("=> ПРЕДУПРЕДУВАЊЕ: loss изгледа нестабилен (NaN или нагол скок). Провери learning rate.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--history_file", default=None)
    args = parser.parse_args()
    plot(args.method, args.history_file)
