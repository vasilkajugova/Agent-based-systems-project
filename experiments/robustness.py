"""
Чисти perturbation функции за robustness студијата (experiments/robustness_study.py).

Секоја функција работи на ЕДНА опсервација (не листа по агент) - самото
"применување на сите агенти во листата" го прави make_perturb_fn() подолу,
кое враќа готова `states -> states` функција за evaluate.py::evaluate(perturb_fn=...).

Методолошки важно: `add_kinematics_noise`/`blur_pixels`/`darken_pixels`
НИКОГАШ не се употребуваат при тренирање - секој модел (kinematics/
pixels/fusion, IQL/VDN) е тренираат ИСКЛУЧИВО на чисти (clean)
опсервации за нив; ова е чисто eval-time манипулација која симулира
несовршени сензори (шумни GPS/радар очитувања, магла/дожд/самрак за
камерата) - НЕ е дел од MDP-то со кое агентот учел. Единствен исклучок е
`drop_branch` (увезена од `agents/networks.py`, не дефинирана тука) -
таа СЕ реупотребува и во train.py за "modality dropout" регуларизација
за време на тренирањето (види коментар кај неа за целосно објаснување).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from agents.networks import FusionState, drop_branch


def add_kinematics_noise(kin: np.ndarray, sigma: float) -> np.ndarray:
    """
    Гаусов шум N(0, sigma^2) на континуираните Kinematics колони (x, y,
    vx, vy, cos_h, sin_h - колони 1..6 во reshape(-1,7)) - симулира шумни
    GPS/радар очитувања на веќе-детектирани возила. Presence-битот
    (колона 0) НЕ се допира - тоа е квалитативно поинаков вид грешка
    (halucinирано/испуштено возило), не сензорска непрецизност на
    позиција/брзина. Опсервацијата е normalize=True (опсег [-1, 1]), па
    clip-увам по додавањето шум за да остане во истиот опсег што мрежата
    го "очекува".
    """
    mat = kin.reshape(-1, 7).copy()
    noise = np.random.normal(0.0, sigma, size=mat[:, 1:].shape)
    mat[:, 1:] = np.clip(mat[:, 1:] + noise, -1.0, 1.0)
    return mat.flatten().astype(np.float32)


def blur_pixels(img: np.ndarray, sigma: float) -> np.ndarray:
    """
    Gaussian blur по просторните (H, W) оски - симулира матна/нефокусирана
    камера (магла, дожд на леќата, дефокус). sigma=0 на стек-оската
    (axis 0, редоследот на фрејмовите) - НАМЕРНО не ги мешам различните
    временски фрејмови еден со друг, само просторно го заматувам секој
    посебно (инаку блур-от би внел и вештачко "temporal smearing" кое
    воопшто не е дел од тоа што сакам да го симулирам тука).
    """
    blurred = gaussian_filter(img.astype(np.float32), sigma=(0, sigma, sigma))
    return np.clip(blurred, 0, 255).astype(np.uint8)


def darken_pixels(img: np.ndarray, factor: float) -> np.ndarray:
    """
    Ги множи интензитетите на пикселите со `factor` (< 1.0 = потемно) -
    симулира вечер/самрак/слаба осветленост на камерата. factor=1.0 е
    no-op (идентитет), factor=0.0 би значело целосно црна слика.
    """
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


# Кои услови се СМИСЛЕНИ за кој obs_mode - kin_noise нема смисла кај
# "pixels" (нема kinematics гранка воопшто), pixel_blur/dark нема смисла
# кај "kinematics", drop_kin/drop_img имаат смисла САМО кај "fusion" (кај
# kinematics-only/pixels-only "гасење на единствената гранка" не е
# деградација туку целосно "слепило" - не е информативен тест).
APPLICABLE_CONDITIONS = {
    "kinematics": ["kin_noise"],
    "pixels": ["pixel_blur", "pixel_dark"],
    "fusion": ["kin_noise", "pixel_blur", "pixel_dark", "drop_kin", "drop_img"],
}


def make_perturb_fn(kind: str, strength: float):
    """
    Фабрика: враќа `states (листа по агент) -> states` функција, готова за
    evaluate.py::evaluate(perturb_fn=...). `kind` е едно од клучевите во
    APPLICABLE_CONDITIONS (без "clean"). `strength` е sigma
    (kin_noise/pixel_blur) или factor (pixel_dark) - се игнорира за
    drop_kin/drop_img (немаат "јачина", гранката или е таму или не е).
    """

    def _apply_one(state):
        if kind == "kin_noise":
            if isinstance(state, FusionState):
                return FusionState(kin=add_kinematics_noise(state.kin, strength), img=state.img)
            return add_kinematics_noise(state, strength)
        if kind == "pixel_blur":
            if isinstance(state, FusionState):
                return FusionState(kin=state.kin, img=blur_pixels(state.img, strength))
            return blur_pixels(state, strength)
        if kind == "pixel_dark":
            if isinstance(state, FusionState):
                return FusionState(kin=state.kin, img=darken_pixels(state.img, strength))
            return darken_pixels(state, strength)
        if kind == "drop_kin":
            return drop_branch(state, "kin")
        if kind == "drop_img":
            return drop_branch(state, "img")
        raise ValueError(f"Непознат perturbation kind: {kind!r}")

    def perturb(states: list) -> list:
        return [_apply_one(s) for s in states]

    return perturb
