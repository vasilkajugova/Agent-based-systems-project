"""
VDN (Value Decomposition Networks) - ова е мојот CTDE пристап
(Centralized Training, Decentralized Execution), директно спомнат во
теоријата на предметот (белешка 10, делот за "MARL алгоритми/семејства...
VDN и QMIX - decomposition на заедничката вредност за кооперативни
тимови").

Идејата накратко:
  - Секој агент i си има своја локална Q-функција Q_i(o_i, a_i), која ја
    пресметува САМО од сопствената опсервација (o_i) - ова е
    decentralized execution, значи при извршување секој агент одлучува
    сам, без да ги гледа опсервациите на другите.
  - Но при ТРЕНИРАЊЕТО (centralized training - тука сум "централизирана"
    јас, скриптата што го контролира тренингот), заедничката TD-цел ја
    пресметувам преку СУМА на сите индивидуални Q-вредности:

        Q_tot(o, a) = Q_1(o_1,a_1) + Q_2(o_2,a_2) + ... + Q_n(o_n,a_n)

    и целата таа сума ја учам спрема заедничката (сумирана) награда.
  - На тој начин, секој агент имплицитно учи да придонесе кон заедничкиот
    успех на тимот, иако при извршување воопшто не ги гледа туѓите
    опсервации.

Ова е поедноставна алтернатива на QMIX (кој додава посложена, нелинеарна
"monotonic" mixing мрежа), но е концептуално чиста и доволно добра за
ниво на проект по овој предмет - јасно ја покажува CTDE идејата од
белешките, без дополнителна сложеност што ќе биде тешко да се објасни/брани.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.networks import QNetwork, DuelingQNetwork, MultiAgentReplayBuffer


class VDNAgent:
    def __init__(
        self,
        num_agents: int,
        obs_dim: int,
        num_actions: int,
        learning_rate: float = 1e-3,
        discount_factor: float = 0.95,
        batch_size: int = 64,
        buffer_size: int = 50_000,
        dueling: bool = False,
        hidden: int = 128,
        device: str | None = None,
    ):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.gamma = discount_factor
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        net_cls = DuelingQNetwork if dueling else QNetwork
        # секој агент си има своја одделна мрежа (тежините НЕ се
        # споделени), но сите заедно се тренираат преку сумираната TD-цел подолу
        self.models = [net_cls(obs_dim, num_actions, hidden).to(self.device) for _ in range(num_agents)]
        self.target_models = [net_cls(obs_dim, num_actions, hidden).to(self.device) for _ in range(num_agents)]
        self.update_target_model()

        # еден единствен optimizer за СИТЕ агентски мрежи заедно - бидејќи
        # ги тренирам преку заедничка (сумирана) loss функција
        params = [p for m in self.models for p in m.parameters()]
        self.optimizer = optim.Adam(params, lr=learning_rate)
        # Huber loss наместо MSE - во VDN Q_tot е СУМА од n индивидуални
        # Q-вредности, значи има поголема магнитуда отколку кај обичен
        # single-agent DQN, па е уште поосетлив на големи TD-грешки.
        # Huber loss (исто како во dqn_agent.py) го ублажува тоа.
        self.criterion = nn.SmoothL1Loss()
        self.memory = MultiAgentReplayBuffer(num_agents, buffer_size)

    def update_target_model(self):
        for tgt, src in zip(self.target_models, self.models):
            tgt.load_state_dict(src.state_dict())

    def update_memory(self, states, actions, rewards, next_states, dones):
        self.memory.push(states, actions, rewards, next_states, dones)

    def get_actions(self, states: list[np.ndarray], epsilon: float) -> list[int]:
        # секој агент одлучува ЗАСЕБНО, само врз основа на сопствената
        # опсервација - ова е decentralized execution делот од CTDE
        actions = []
        for i, state in enumerate(states):
            if np.random.rand() < epsilon:
                actions.append(np.random.randint(self.num_actions))
            else:
                state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    q = self.models[i](state_t)
                actions.append(int(torch.argmax(q, dim=1).item()))
        return actions

    def train_step(self) -> float | None:
        if len(self.memory) < self.batch_size:
            return None
        batch = self.memory.sample(self.batch_size)

        q_tot = torch.zeros(self.batch_size, device=self.device)
        target_tot = torch.zeros(self.batch_size, device=self.device)

        for i in range(self.num_agents):
            states = torch.tensor(batch["states"][i], dtype=torch.float32, device=self.device)
            actions = torch.tensor(batch["actions"][i], dtype=torch.long, device=self.device).unsqueeze(1)
            next_states = torch.tensor(batch["next_states"][i], dtype=torch.float32, device=self.device)
            rewards_i = torch.tensor(batch["rewards"][i], dtype=torch.float32, device=self.device)
            dones_i = torch.tensor(batch["dones"][i], dtype=torch.float32, device=self.device)

            # Q_i(o_i, a_i) за агент i, па го собирам во заедничкиот Q_tot (централизирано тренирање)
            q_i = self.models[i](states).gather(1, actions).squeeze(1)
            q_tot = q_tot + q_i

            # ВАЖЕН детаљ на кој наидов при тестирање, па сакам добро да го
            # објаснам: во моето сценарио секој агент завршува (се судира
            # или стигнува до целта) во СВОЕ индивидуално време - агент 1
            # може да заврши на чекор 3, додека агент 2 сè уште е активен
            # до чекор 8 (тимската епизода трае додека НЕ завршат СИТЕ
            # агенти - види train.py). Во првата верзија на кодот ова
            # погрешно го третирав со ЕДЕН заеднички "any_done" флаг, кој
            # го НУЛИРАШЕ bootstrap-от (идната вредност) за ЦЕЛИОТ Q_tot
            # штом БИЛО КОЈ агент е done - иако другите агенти сè уште
            # имаат валидна идна вредност! Тоа систематски ја зголемуваше
            # TD-грешката и внесуваше шум во тренингот (го забележав како
            # нестабилна/шумна collision-rate крива). Поправката: секој
            # агент го маскира САМО својот сопствен bootstrap член со
            # (1 - done_i), па дури потоа ги собирам сите - ова е
            # стандардната VDN декомпозиција:
            #     Q_tot(target) = Σ_i [ r_i + gamma * Q_i'(next) * (1 - done_i) ]
            with torch.no_grad():
                next_q_i = self.target_models[i](next_states).max(dim=1).values
                target_i = rewards_i + self.gamma * next_q_i * (1 - dones_i)
            target_tot = target_tot + target_i

        loss = self.criterion(q_tot, target_tot)
        self.optimizer.zero_grad()
        loss.backward()
        # gradient clipping низ СИТЕ агентски мрежи заедно (истата причина
        # како кај DQNAgent - спречува нестабилно/"експлозивно" учење,
        # тука уште поважно бидејќи сите мрежи се тренираат заедно)
        torch.nn.utils.clip_grad_norm_(
            [p for m in self.models for p in m.parameters()], max_norm=10.0
        )
        self.optimizer.step()
        return float(loss.item())

    def save(self, path_prefix: str):
        for i, m in enumerate(self.models):
            torch.save(m.state_dict(), f"{path_prefix}_agent{i}.pt")

    def load(self, path_prefix: str):
        for i, m in enumerate(self.models):
            m.load_state_dict(torch.load(f"{path_prefix}_agent{i}.pt", map_location=self.device))
        self.update_target_model()
