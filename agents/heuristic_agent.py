"""
Rule-based (правило-базиран) baseline: нема учење воопшто, само едно
едноставно правило - "успори ако има возило блиску пред тебе, инаку
забрзувај". Ова е долната референтна точка во споредбата - без baseline,
не можам да знам дали RL агентите реално придонесуваат нешто, или само
"случајно" изгледаат добро (ова е точно поентата од белешка 13 од
предметот: "Без baseline не се знае дали RL навистина помага").

Опсервацијата (за секој агент) е "спласната" (flatten-ирана) Kinematics
матрица: vehicles_count х 7 колони со редослед [presence, x, y, vx, vy,
cos_h, sin_h]. Првиот "ред" (првите 7 броја) СЕКОГАШ е самото возило
(себеси), а остатокот се најблиските други возила во релативни координати
(гледано од перспектива на самото возило).

Познато ограничување (намерно, не е грешка): правилово гледа само
СТАТИЧКО растојание до најблиското возило, не и релативна брзина или
"време до судир" (time-to-collision). Значи не прави разлика помеѓу
"возило далеку, но многу брзо се приближува" и "возило далеку, стои
мирно" - и двете добиваат иста FASTER одлука. Ова е свесен избор за да
остане навистина едноставен и лесно разбирлив baseline (спротивно на RL
агентите, кои се "црна кутија"). Вреди да се спомене во дискусијата на
извештајот зошто heuristic-от понекогаш реагира "доцна" во споредба со
RL агентите.
"""
from __future__ import annotations

import numpy as np


class HeuristicAgent:
    ACTIONS = {"SLOWER": 0, "IDLE": 1, "FASTER": 2}

    def __init__(self, danger_distance: float = 0.15):
        # Внимание: опсервацијата е НОРМАЛИЗИРАНА (normalize=True во
        # envs/multi_agent_intersection.py), значи x/y се мапирани од
        # опсегот [-100,100] метри во [-1,1]. Затоа danger_distance е во
        # нормализирани единици, не во метри (0.15 нормализирано ≈ 15
        # метри во реалниот свет).
        self.danger_distance = danger_distance

    def get_action(self, obs_flat: np.ndarray) -> int:
        obs = obs_flat.reshape(-1, 7)  # враќам ја матрицата (vehicles_count, [presence,x,y,vx,vy,cos_h,sin_h])
        others = obs[1:]  # редот 0 е самото возило, останатите се другите возила
        present = others[:, 0] > 0.5  # presence е 0/1 - земам само возилата што реално постојат
        if not present.any():
            return self.ACTIONS["FASTER"]  # нема никој наоколу, слободно забрзувај

        rel_x = others[present, 1]
        rel_y = others[present, 2]
        dist = np.sqrt(rel_x**2 + rel_y**2)  # обично евклидово растојание до секое возило
        min_dist = dist.min() if len(dist) else np.inf

        # едноставно правило со два прага: многу блиску -> забави, средно
        # блиску -> одржи брзина, инаку -> забрзај
        if min_dist < self.danger_distance * 0.5:
            return self.ACTIONS["SLOWER"]
        elif min_dist < self.danger_distance:
            return self.ACTIONS["IDLE"]
        return self.ACTIONS["FASTER"]

    def get_actions(self, states: list[np.ndarray], epsilon: float = 0.0) -> list[int]:
        # epsilon се игнорира тука, ова е детерминистичко правило, нема
        # учење, значи нема потреба од epsilon-greedy истражување. Го
        # задржувам параметарот само за да може да се повикува со ист
        # интерфејс како IQL/VDN агентите во train.py/evaluate.py.
        return [self.get_action(s) for s in states]
