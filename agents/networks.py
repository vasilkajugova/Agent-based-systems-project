"""
Тука се основните "градбени блокови" што ги користам во сите агенти:
Q-мрежа (со опционален dueling "head"), и replay buffer (меморија на
претходни искуства).

Ова е директна надградба на deep_q_learning_blank.py од лабораториските
вежби (Лаб 3/4) - истата DQN/Dueling DQN логика што ја работевме,
само дообработена и приспособена за мулти-агентен контекст (VDN mixer-от
подоцна ги комбинира индивидуалните Q-вредности од овие мрежи).
"""
from __future__ import annotations

import random
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn

# Со namedtuple е полесно да си играм со податоците подоцна - state.action
# наместо state[1], и веднаш се гледа кое поле што значи.
Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


class QNetwork(nn.Module):
    """
    Обична MLP (multi-layer perceptron) Q-мрежа - на влез ја зема
    опсервацијата (state), а на излез дава по едно Q-вредност за секоја
    можна акција: Q(s, ·).

    Архитектурата е намерно едноставна (2 скриени слоја од по 128
    неврони + ReLU) - за овој проект не ми требаше нешто посложено,
    важно е принципот да работи правилно.
    """

    def __init__(self, obs_dim: int, num_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DuelingQNetwork(nn.Module):
    """
    Dueling архитектура - истото ова го работевме и на предметот. Идејата
    е дека Q-вредноста може да се разложи на два дела:

        Q(s,a) = V(s) + (A(s,a) - просек_a A(s,a))

    каде V(s) е "колку добра е самата состојба" (без разлика на акцијата),
    а A(s,a) е "предност/advantage" - колку е таа конкретна акција подобра
    или полоша од просекот на акциите во таа состојба. Ова помага мрежата
    побрзо да научи кога состојбата е "добра"/"лоша" независно од тоа која
    акција точно ќе се избере.
    """

    def __init__(self, obs_dim: int, num_actions: int, hidden: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.value = nn.Linear(hidden, 1)
        self.advantage = nn.Linear(hidden, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.shared(x)
        v = self.value(h)
        a = self.advantage(h)
        return v + (a - a.mean(dim=1, keepdim=True))


class ReplayBuffer:
    """
    Стандардна "меморија на искуства" (experience replay) - секогаш кога
    агентот направи чекор во околината, го зачувувам тоа искуство овде, а
    подоцна при тренирање земам случаен batch (група) искуства оттука
    наместо да учам само од последниот чекор. Ова е важно затоа што
    последователните чекори во една епизода се многу слични/поврзани
    (correlated), а невронските мрежи учат подобро кога примероците се
    поразновидни/случајни - точно затоа постои replay buffer, го
    работевме и на предметот при DQN.

    Одлучив да го имплементирам со обична Python листа + circular index
    (значи кога буферот е полн, најстариот запис се презапишува), а НЕ со
    `collections.deque`. Причината: sample() бара случаен пристап по
    индекс до секој елемент од batch-от (random.sample), а обичната листа
    дава тоа во O(1) (веднаш), додека deque е поблиску до "поврзана листа"
    каде пристап по индекс чини O(n) (толку побавно колку е подалеку од
    крајот). За buffer_size=50 000 и train_step() кој се повикува на СЕКОЈ
    чекор од тренингот, оваа разлика реално се чувствува во брзината.
    """

    def __init__(self, capacity: int = 50_000):
        self.capacity = capacity
        self.buffer: list = []
        self._pos = 0

    def push(self, state, action, reward, next_state, done):
        transition = Transition(state, action, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self._pos] = transition
        self._pos = (self._pos + 1) % self.capacity

    def sample(self, batch_size: int) -> Transition:
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)


class MultiAgentReplayBuffer:
    """
    Ист принцип како ReplayBuffer погоре, само прилагодено за мулти-
    агентен случај - тука зачувувам "заеднички" (joint) чекори: за секој
    временски чекор ги паметам state_i и action_i на СИТЕ агенти
    истовремено, не одделно за секој агент.

    Ова ми е потребно конкретно за VDN (CTDE методот): mixer-от мора да ги
    види Q-вредностите на СИТЕ агенти во ИСТ момент за да ја пресмета
    заедничката цел (Q_tot) - не можам да земам случаен чекор за агент 1 и
    друг случаен чекор за агент 2, мора да е ист временски момент.

    Истата O(1) circular-list имплементација како ReplayBuffer погоре
    (наместо deque) - иста причина: побрзо random-access sampling.
    """

    def __init__(self, num_agents: int, capacity: int = 50_000):
        self.num_agents = num_agents
        self.capacity = capacity
        self.buffer: list = []
        self._pos = 0

    def push(self, states, actions, rewards, next_states, dones):
        # states/actions/... се листи со должина num_agents (еден запис по агент)
        item = (states, actions, rewards, next_states, dones)
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            self.buffer[self._pos] = item
        self._pos = (self._pos + 1) % self.capacity

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        # ги "преорганизирам" податоците од "по чекор" во "по агент": за секој
        # агент правам посебен batch, за агентот подоцна да може да го обработи со својата мрежа
        out = {"states": [], "actions": [], "rewards": [], "next_states": [], "dones": []}
        for i in range(self.num_agents):
            out["states"].append(np.stack([s[i] for s in states]))
            out["actions"].append(np.array([a[i] for a in actions]))
            out["rewards"].append(np.array([r[i] for r in rewards], dtype=np.float32))
            out["next_states"].append(np.stack([ns[i] for ns in next_states]))
            out["dones"].append(np.array([d[i] for d in dones], dtype=np.float32))
        return out

    def __len__(self):
        return len(self.buffer)
