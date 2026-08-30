"""
Тука се основните "градбени блокови" што ги користам во сите агенти:
Q-мрежа (со опционален dueling "head"), и replay buffer (меморија на
претходни искуства).

Ова е директна надградба на deep_q_learning_blank.py од лабораториските
вежби (Лаб 3/4) - истата DQN/Dueling DQN логика што ја работевме,
само дообработена и приспособена за мулти-агентен контекст (VDN mixer-от
подоцна ги комбинира индивидуалните Q-вредности од овие мрежи).

Проширување за 3-те observation режими (Kinematics / Pixels / Fusion):
опсервацијата по агент е СЕГА или обичен numpy вектор (kinematics/pixels
режим), или `FusionState(kin, img)` (fusion режим - двa одделни "гранки").
Наместо да го "растурам" ова obs_mode-разликување низ секој агент
(dqn_agent.py, vdn_agent.py) одделно, го centraliziram тука преку неколку
мали helper функции (`stack_states`, `to_torch`, `to_batch_tensor`,
`forward_q`) - секој агент само ги повикува нив, без сам да мора да знае
"дали ова е FusionState или обичен тензор". Единственото место кое РЕАЛНО
одлучува која Q-мрежа/архитектура да се употреби е `build_q_network()`
подолу.
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

# Опсервација по агент за Fusion режимот: kin = flatten Kinematics вектор
# (исто како кинематскиот режим), img = стек grayscale фрејмови (стек од
# `pixel_stack_size` кадри, секој (H, W), agent-centric - види
# envs/multi_agent_intersection.py). И двете гранки ги гледа истиот агент
# во ИСТ временски чекор, само низ различни "сетила".
FusionState = namedtuple("FusionState", ["kin", "img"])


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


class _CNNEncoder(nn.Module):
    """
    Заедничка CNN "труба" за Pixel/Fusion мрежите подолу - 3 конволутивни
    слоја (16->32->32 канали, kernel 5/3/3, stride 2) + FC глава со
    `out_dim` неврони. Намерно плитка/мала (иста филозофија како 128-
    неврони MLP-от погоре - за ова сценарио не ми треба ResNet длабочина,
    сликите се мали 64x64 agent-centric кадри, не 84x84 Atari фрејмови со
    комплексна сцена).

    Влезот е стек grayscale фрејмови (stack_size, H, W), вредности во
    [0, 255] (uint8/float, како што ги враќа highway-env-овата
    GrayscaleObservation) - ги нормализирам на [0, 1] ТУКА, во forward(),
    за агентите (dqn_agent.py/vdn_agent.py) воопшто да не мора да мислат
    за нормализација на сликата (истото како kinematics-опсервацијата, која
    веќе доаѓа преднормализирана од envs/multi_agent_intersection.py преку
    normalize=True).
    """

    def __init__(self, img_shape: tuple[int, int, int], out_dim: int = 128):
        super().__init__()
        in_ch, h, w = img_shape
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.ReLU(),
        )
        with torch.no_grad():
            conv_out_dim = self.conv(torch.zeros(1, in_ch, h, w)).flatten(1).shape[1]
        self.fc = nn.Sequential(nn.Linear(conv_out_dim, out_dim), nn.ReLU())

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x = img.float() / 255.0
        return self.fc(self.conv(x).flatten(1))


class PixelQNetwork(nn.Module):
    """
    CNN Q-мрежа за Pixels-only режимот: влез е стек agent-centric grayscale
    фрејмови (не kinematics вектор) - секој агент "гледа" само сопствена,
    локална слика центрирана на себе (види `_pixel_obs`/`observer_vehicle`
    во envs/multi_agent_intersection.py), не глобален screenshot на цела
    раскрсница споделен меѓу сите.
    """

    def __init__(self, img_shape: tuple[int, int, int], num_actions: int, hidden: int = 128):
        super().__init__()
        self.encoder = _CNNEncoder(img_shape, out_dim=hidden)
        self.head = nn.Linear(hidden, num_actions)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(img))


class DuelingPixelQNetwork(nn.Module):
    """Dueling варијанта на PixelQNetwork - иста V(s)+A(s,a) идеја како DuelingQNetwork погоре, само CNN енкодер наместо MLP."""

    def __init__(self, img_shape: tuple[int, int, int], num_actions: int, hidden: int = 128):
        super().__init__()
        self.encoder = _CNNEncoder(img_shape, out_dim=hidden)
        self.value = nn.Linear(hidden, 1)
        self.advantage = nn.Linear(hidden, num_actions)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        h = self.encoder(img)
        v = self.value(h)
        a = self.advantage(h)
        return v + (a - a.mean(dim=1, keepdim=True))


class FusionQNetwork(nn.Module):
    """
    Fusion Q-мрежа: MLP гранка на локалниот kinematics вектор + CNN гранка
    на локалната grayscale слика, СПОЕНИ (concat) пред финалната FC глава -
    точно архитектурата побарана за проектот. Двете гранки учат ОДДЕЛНИ,
    комплементарни претстави (kinematics: прецизни бројки за позиција/
    брзина на најблиските возила; pixels: пошироко просторно/визуелно
    сценарио), а само финалната глава учи КАКО да ги комбинира.
    """

    def __init__(self, obs_dim: int, img_shape: tuple[int, int, int], num_actions: int, hidden: int = 128):
        super().__init__()
        self.kin_branch = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU())
        self.img_branch = _CNNEncoder(img_shape, out_dim=hidden)
        self.head = nn.Linear(hidden * 2, num_actions)

    def forward(self, kin: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        h = torch.cat([self.kin_branch(kin), self.img_branch(img)], dim=1)
        return self.head(h)


class DuelingFusionQNetwork(nn.Module):
    """Dueling варијанта на FusionQNetwork."""

    def __init__(self, obs_dim: int, img_shape: tuple[int, int, int], num_actions: int, hidden: int = 128):
        super().__init__()
        self.kin_branch = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU())
        self.img_branch = _CNNEncoder(img_shape, out_dim=hidden)
        self.value = nn.Linear(hidden * 2, 1)
        self.advantage = nn.Linear(hidden * 2, num_actions)

    def forward(self, kin: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        h = torch.cat([self.kin_branch(kin), self.img_branch(img)], dim=1)
        v = self.value(h)
        a = self.advantage(h)
        return v + (a - a.mean(dim=1, keepdim=True))


def build_q_network(
    obs_mode: str,
    obs_dim: int | None,
    img_shape: tuple[int, int, int] | None,
    num_actions: int,
    dueling: bool,
    hidden: int = 128,
) -> nn.Module:
    """
    Единствено место кое одлучува која класа Q-мрежа да се инстанцира,
    врз основа на obs_mode - DQNAgent/VDNAgent само го викаат ова, наместо
    секој одделно да го копира истиот if/elif врз obs_mode.
    """
    if obs_mode == "kinematics":
        return DuelingQNetwork(obs_dim, num_actions, hidden) if dueling else QNetwork(obs_dim, num_actions, hidden)
    if obs_mode == "pixels":
        return (
            DuelingPixelQNetwork(img_shape, num_actions, hidden)
            if dueling
            else PixelQNetwork(img_shape, num_actions, hidden)
        )
    if obs_mode == "fusion":
        return (
            DuelingFusionQNetwork(obs_dim, img_shape, num_actions, hidden)
            if dueling
            else FusionQNetwork(obs_dim, img_shape, num_actions, hidden)
        )
    raise ValueError(f"Непознат obs_mode: {obs_mode!r} (очекувам 'kinematics'/'pixels'/'fusion')")


def stack_states(states: list) -> np.ndarray | FusionState:
    """
    Ги "стакува" листа опсервации (по еден state по batch-елемент, секој
    или numpy вектор/слика или FusionState) во ЕДЕН numpy array
    (kinematics/pixels режим), или во FusionState со ДВЕ одделно стакувани
    numpy array-и (fusion режим) - обичен np.stack() директно врз листа
    FusionState-и НЕ работи (namedtuple од array-и не е самиот array-like).

    Го користат и MultiAgentReplayBuffer.sample() (batch по агент, VDN) и
    DQNAgent.train_step() (batch за еден IQL агент).
    """
    if isinstance(states[0], FusionState):
        return FusionState(
            kin=np.stack([s.kin for s in states]),
            img=np.stack([s.img for s in states]),
        )
    return np.stack(states)


def to_torch(state, device) -> torch.Tensor | FusionState:
    """
    Го претвора еден (веќе-стакуван batch, или единечен) state во torch
    tensor(и), float32 - за FusionState враќа FusionState од ДВА тензори
    (kin, img). Нормализацијата на сликата е ВНАТРЕ во _CNNEncoder.forward
    погоре, не тука - ова е само генерички numpy -> torch чекор.
    """
    if isinstance(state, FusionState):
        return FusionState(
            kin=torch.tensor(state.kin, dtype=torch.float32, device=device),
            img=torch.tensor(state.img, dtype=torch.float32, device=device),
        )
    return torch.tensor(state, dtype=torch.float32, device=device)


def to_batch_tensor(state, device) -> torch.Tensor | FusionState:
    """
    Исто како to_torch(), само дополнително додава batch димензија
    (unsqueeze(0)) - за ЕДЕН единствен state (не веќе-стакуван batch од
    replay buffer-от), како во get_action()/get_actions() кога агентот
    одлучува за само една моментална опсервација.
    """
    t = to_torch(state, device)
    if isinstance(t, FusionState):
        return FusionState(kin=t.kin.unsqueeze(0), img=t.img.unsqueeze(0))
    return t.unsqueeze(0)


def forward_q(model: nn.Module, state_t) -> torch.Tensor:
    """
    Го повикува model(...) со точниот број аргументи - FusionState носи
    ДВЕ гранки (model(kin, img)), обичен тензор носи ЕДНА (model(x)).
    Централизирано тука за агентите (dqn_agent.py/vdn_agent.py) да не
    мораат сами да проверуваат obs_mode на секое место каде повикуваат
    Q-мрежа (get_action, train_step Q(s,a), train_step target/double-DQN
    гранка - 3-4 места по агент, лесно за пропуштање едно ако е рачно).
    """
    if isinstance(state_t, FusionState):
        return model(state_t.kin, state_t.img)
    return model(state_t)


def drop_branch(state: FusionState, which: str) -> FusionState:
    """
    За Fusion опсервација, ја "гаси" (нулира) едната гранка. Двe одделни
    употреби во проектот:
      1. Eval-time robustness perturbation (experiments/robustness.py) -
         симулира привремен дефект на еден сензор.
      2. Training-time "modality dropout" регуларизација (train.py::run_training,
         modality_dropout_prob>0) - откриено емпириски дека Fusion мрежата,
         тренирана НА ОБИЧЕН начин, речиси целосно ја игнорира сликата и се
         потпира само на kinematics (Q-вредностите остануваат речиси
         непроменети дури и кога сликата е целосно црна - "modality
         collapse", потврдено на сите 30 (метод×seed×агент) комбинации).
         Повремено гасење на kin-гранката ЗА ВРЕМЕ на тренирањето го
         принудува агентот понекогаш да одлучува САМО од сликата, па
         градиентот кон CNN-гранката веќе не може да остане "мртов".
    Функцијата е чиста (не мутира state), затоа е безбедна за реупотреба
    и за двете намени. `which` е "kin" или "img" - другата гранка
    останува НЕДОПРЕНА.
    """
    if which == "kin":
        return FusionState(kin=np.zeros_like(state.kin), img=state.img)
    if which == "img":
        return FusionState(kin=state.kin, img=np.zeros_like(state.img))
    raise ValueError(f"which мора да е 'kin' или 'img', добив: {which!r}")


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
        #
        # stack_states() наместо гол np.stack() - за kinematics/pixels режим
        # е идентично (np.stack на листа numpy array-и), но за fusion режим
        # секој state е FusionState(kin, img) и np.stack() директно врз
        # листа namedtuple-и не работи (види agents/networks.py::stack_states).
        out = {"states": [], "actions": [], "rewards": [], "next_states": [], "dones": []}
        for i in range(self.num_agents):
            out["states"].append(stack_states([s[i] for s in states]))
            out["actions"].append(np.array([a[i] for a in actions]))
            out["rewards"].append(np.array([r[i] for r in rewards], dtype=np.float32))
            out["next_states"].append(stack_states([ns[i] for ns in next_states]))
            out["dones"].append(np.array([d[i] for d in dones], dtype=np.float32))
        return out

    def __len__(self):
        return len(self.buffer)
