"""
English-labeled variant of watch.py. Re-runs the seed=1 demo episode with
the same trained models and the same env seed as the original, with the
overlay text translated to English.

CAUTION - not guaranteed bit-for-bit identical to the original run: the
env seed fixes the traffic scenario, but per-step Q-value ties can resolve
differently depending on PyTorch/BLAS floating-point execution order, so a
re-run can diverge in exact rewards/outcomes (confirmed once: the same
qualitative outcome per agent, but different reward values, when this was
re-run for the thesis translation). If you regenerate
watch_seed1_lastframe.png this way for use in the thesis, re-check its
numbers against report/Thesis_MultiAgentRL_Intersection_EN.docx (Section
5.5) before embedding -- do not assume they still match. The actual
English figure currently embedded in the thesis was produced by directly
translating the overlay text on the original (Macedonian) last frame
in-place with Pillow, not by re-running this script.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from envs.multi_agent_intersection import MultiAgentIntersectionEnv
from agents.iql_manager import IQLManager
from agents.vdn_agent import VDNAgent
from agents.heuristic_agent import HeuristicAgent

RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures_en"
FIGURES_DIR.mkdir(exist_ok=True, parents=True)

IDLE_ACTION = 1
ACTION_NAMES = {0: "SLOWER", 1: "IDLE", 2: "FASTER"}

METHOD_ORDER = ["heuristic", "iql", "vdn"]
METHOD_LABELS = {
    "heuristic": "Heuristic (rule-based)",
    "iql": "IQL (independent agents)",
    "vdn": "VDN (CTDE)",
}
METHOD_COLORS = {
    "heuristic": (127, 140, 141),
    "iql": (230, 126, 34),
    "vdn": (41, 128, 185),
}
STATUS_COLORS = {
    "ACTIVE": (46, 204, 113),
    "CRASHED": (231, 76, 60),
    "ARRIVED": (52, 152, 219),
    "DONE": (149, 165, 166),
}

PANEL_W, PANEL_H = 420, 420
TITLE_H = 34
OVERLAY_H_PER_AGENT = 20
HEADER_H = 30


def _load_font(size: int):
    for candidate in ("arial.ttf", "C:/Windows/Fonts/arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            continue
    return ImageFont.load_default()


FONT_TITLE = _load_font(18)
FONT_BODY = _load_font(13)
FONT_HEADER = _load_font(15)


def find_model_prefix(method: str, seed: int) -> str | None:
    if (RESULTS_DIR / f"{method}_model_agent0.pt").exists():
        return str(RESULTS_DIR / f"{method}_model")
    tagged = RESULTS_DIR / f"{method}_seed{seed}_model"
    if (RESULTS_DIR / f"{method}_seed{seed}_model_agent0.pt").exists():
        return str(tagged)
    any_match = sorted(RESULTS_DIR.glob(f"{method}_seed*_model_agent0.pt"))
    if any_match:
        return str(any_match[0]).replace("_agent0.pt", "")
    return None


class MethodPanel:
    def __init__(self, method: str, num_agents: int, seed: int):
        self.method = method
        self.num_agents = num_agents
        self.available = True
        self.skip_reason = None

        model_prefix = None
        if method != "heuristic":
            model_prefix = find_model_prefix(method, seed)
            if model_prefix is None:
                self.available = False
                self.skip_reason = f"no saved model for '{method}' in results/ - skipped"
                return

        self.env = MultiAgentIntersectionEnv(
            num_agents=num_agents, render_mode="rgb_array", config_overrides={"scaling": 3.5},
        )
        if method == "heuristic":
            self.manager = HeuristicAgent()
        elif method == "iql":
            self.manager = IQLManager(num_agents, self.env.obs_dim, self.env.num_actions, double_dqn=True)
            self.manager.load(model_prefix)
        elif method == "vdn":
            self.manager = VDNAgent(num_agents, self.env.obs_dim, self.env.num_actions, dueling=True)
            self.manager.load(model_prefix)
        else:
            raise ValueError(method)

        self.states = None
        self.active = np.ones(num_agents, dtype=bool)
        self.status = ["ACTIVE"] * num_agents
        self.cum_reward = np.zeros(num_agents)
        self.last_actions = [None] * num_agents
        self.done_all = False
        self.last_frame = None

    def reset(self, seed: int):
        if not self.available:
            return
        self.states = self.env.reset(seed=seed)
        self.active = np.ones(self.num_agents, dtype=bool)
        self.status = ["ACTIVE"] * self.num_agents
        self.cum_reward = np.zeros(self.num_agents)
        self.last_actions = [None] * self.num_agents
        self.done_all = False
        self.last_frame = self.env.env.render()

    def step(self):
        if not self.available or self.done_all:
            return
        if self.method == "heuristic":
            actions = self.manager.get_actions(self.states)
        else:
            actions = self.manager.get_actions(self.states, epsilon=0.0)
        actions = [a if self.active[i] else IDLE_ACTION for i, a in enumerate(actions)]
        self.last_actions = actions

        next_states, rewards, dones, info = self.env.step(actions)
        self.cum_reward += np.array(rewards)
        dones_arr = np.array(dones)
        newly_done = self.active & dones_arr
        for i in np.where(newly_done)[0]:
            if info["crashed"][i]:
                self.status[i] = "CRASHED"
            elif info["arrived"][i]:
                self.status[i] = "ARRIVED"
            else:
                self.status[i] = "DONE"
        self.active = self.active & ~dones_arr
        self.states = next_states
        self.done_all = all(dones)
        self.last_frame = self.env.env.render()

    def close(self):
        if self.available:
            self.env.close()


def compose_frame(panels: list[MethodPanel], step_idx: int) -> Image.Image:
    n = len(panels)
    header_h = HEADER_H
    max_agents = max(p.num_agents for p in panels if p.available) if any(p.available for p in panels) else 1
    overlay_h = max_agents * OVERLAY_H_PER_AGENT + 8
    panel_total_h = TITLE_H + PANEL_H + overlay_h
    canvas = Image.new("RGB", (n * PANEL_W, header_h + panel_total_h), (20, 20, 24))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), f"Step {step_idx}", font=FONT_HEADER, fill=(230, 230, 230))

    for col, p in enumerate(panels):
        x0 = col * PANEL_W
        y0 = header_h
        color = METHOD_COLORS.get(p.method, (200, 200, 200))
        draw.rectangle([x0, y0, x0 + PANEL_W, y0 + TITLE_H], fill=color)
        draw.text((x0 + 10, y0 + 8), METHOD_LABELS.get(p.method, p.method), font=FONT_TITLE, fill=(255, 255, 255))

        sim_y0 = y0 + TITLE_H
        if not p.available:
            draw.rectangle([x0, sim_y0, x0 + PANEL_W, sim_y0 + PANEL_H], fill=(40, 40, 44))
            draw.text((x0 + 20, sim_y0 + PANEL_H // 2 - 10), p.skip_reason or "unavailable",
                       font=FONT_BODY, fill=(200, 120, 120))
        elif p.last_frame is not None:
            frame_img = Image.fromarray(p.last_frame).resize((PANEL_W, PANEL_H))
            canvas.paste(frame_img, (x0, sim_y0))
            if p.done_all:
                draw.rectangle([x0 + 8, sim_y0 + 8, x0 + 100, sim_y0 + 30], fill=(0, 0, 0))
                draw.text((x0 + 14, sim_y0 + 12), "FINISHED", font=FONT_BODY, fill=(255, 255, 255))

        ov_y0 = sim_y0 + PANEL_H
        draw.rectangle([x0, ov_y0, x0 + PANEL_W, ov_y0 + overlay_h], fill=(28, 28, 32))
        if p.available:
            for i in range(p.num_agents):
                status = p.status[i]
                act = ACTION_NAMES.get(p.last_actions[i], "-") if p.last_actions[i] is not None else "-"
                line = f"A{i}: {act:<6} R={p.cum_reward[i]:+.1f}  {status}"
                draw.text((x0 + 8, ov_y0 + 4 + i * OVERLAY_H_PER_AGENT), line,
                          font=FONT_BODY, fill=STATUS_COLORS.get(status, (220, 220, 220)))
    return canvas


def run_episode(panels: list[MethodPanel], seed: int, max_steps: int = 120) -> list[Image.Image]:
    for p in panels:
        p.reset(seed)
    frames = [compose_frame(panels, 0)]
    step = 0
    while step < max_steps and not all(p.done_all for p in panels if p.available):
        for p in panels:
            p.step()
        step += 1
        frames.append(compose_frame(panels, step))
    return frames


def save_gif(frames: list[Image.Image], out_path: Path, fps: int):
    frames[0].save(
        out_path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0, optimize=True,
    )


def main():
    seed = 1
    num_agents = 3
    print(f"[watch_en] preparing panels for seed={seed} ...")
    panels = [MethodPanel(m, num_agents, seed) for m in METHOD_ORDER]
    for p in panels:
        if not p.available:
            print(f"[watch_en] {p.skip_reason}")

    print(f"[watch_en] recording episode (seed={seed}) ...")
    frames = run_episode(panels, seed=seed, max_steps=120)
    print(f"[watch_en] {len(frames)} frames composed.")

    gif_path = FIGURES_DIR / f"watch_seed{seed}.gif"
    save_gif(frames, gif_path, fps=5)
    print(f"[watch_en] saved: {gif_path}")

    last_frame_path = FIGURES_DIR / f"watch_seed{seed}_lastframe.png"
    frames[-1].save(last_frame_path)
    print(f"[watch_en] saved: {last_frame_path}")

    for p in panels:
        p.close()


if __name__ == "__main__":
    main()
