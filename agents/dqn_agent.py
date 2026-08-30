"""
DQN агент, со опции за Double DQN и Dueling DQN. Ова е надградба од
deep_q_learning_blank.py (лабораториски вежби). Во мојот проект овој
ист агент го користам на 3 начини:

  1. Како single-agent baseline - контрола на само едно возило, за
     споредба со мулти-агентните случаи.
  2. Independent Q-Learning (IQL): по еден вакво независен агент за
     секое возило. Секој агент ги третира сите други возила само како
     дел од околината (која притоа е "нестационарна", бидејќи и другите
     агенти учат и си ја менуваат политиката во исто време) - ова е
     токму нестабилноста опишана во темата "Multi-Agent Systems" од
     предметот (белешка 10): "Од гледна точка на еден агент, средината
     станува понестабилна: другите агенти учат и ја менуваат нивната
     политика."
  3. Градбен елемент на VDN агентот (agents/vdn_agent.py) - истите
     Q-мрежи, само тренирани заедно преку сумирана (mixed) TD-цел
     наместо секоја одделно.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.networks import ReplayBuffer, build_q_network, forward_q, stack_states, to_batch_tensor, to_torch


class DQNAgent:
    def __init__(
        self,
        obs_dim: int | None,
        num_actions: int,
        learning_rate: float = 1e-3,
        discount_factor: float = 0.95,
        batch_size: int = 64,
        buffer_size: int = 50_000,
        double_dqn: bool = False,
        dueling: bool = False,
        hidden: int = 128,
        device: str | None = None,
        obs_mode: str = "kinematics",
        img_shape: tuple[int, int, int] | None = None,
    ):
        # obs_mode/img_shape се нови (Pixels/Fusion проширување) - default
        # вредностите ("kinematics", None) точно го репродуцираат старото
        # однесување, значи постоечките повици (IQL со само obs_dim/
        # num_actions) остануваат непроменети.
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.gamma = discount_factor
        self.batch_size = batch_size
        self.double_dqn = double_dqn
        self.obs_mode = obs_mode
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = build_q_network(obs_mode, obs_dim, img_shape, num_actions, dueling, hidden).to(self.device)
        self.target_model = build_q_network(obs_mode, obs_dim, img_shape, num_actions, dueling, hidden).to(self.device)
        self.update_target_model()

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        # Huber loss (SmoothL1) наместо обична MSE - е помалку чувствителна
        # на големи TD-грешки (outliers). Ова го додадов откако забележав
        # проблем при тестирање: со MSE, loss-от постојано растеше наместо
        # да опаѓа. Huber loss за мали грешки се однесува како MSE, а за
        # големи грешки како MAE (линеарно), па не "експлодира" толку лесно.
        self.criterion = nn.SmoothL1Loss()
        self.memory = ReplayBuffer(buffer_size)

    def update_target_model(self):
        # ја "копирам" тековната мрежа во target мрежата. Target мрежата
        # служи за да се пресметуваат TD-целите - ако ја ажурирам на секој
        # чекор наместо повремено, целта постојано би "бегала" и учењето
        # би било нестабилно (ова го работевме на предметот при DQN).
        self.target_model.load_state_dict(self.model.state_dict())

    def update_memory(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def get_action(self, state, epsilon: float) -> int:
        # epsilon-greedy: со веројатност epsilon бирам случајна акција
        # (истражување), инаку ја бирам најдобрата акција според моделот
        # (искористување). Класична epsilon-greedy стратегија од предметот.
        # state е обичен numpy вектор/слика (kinematics/pixels режим) или
        # FusionState(kin, img) (fusion режим) - to_batch_tensor/forward_q
        # (agents/networks.py) го крие тоа разликување тука.
        if np.random.rand() < epsilon:
            return np.random.randint(self.num_actions)
        state_t = to_batch_tensor(state, self.device)
        with torch.no_grad():
            q_values = forward_q(self.model, state_t)
        return int(torch.argmax(q_values, dim=1).item())

    def train_step(self) -> float | None:
        # Ако сè уште немам доволно искуства во меморијата за цел batch,
        # едноставно прескокнувам чекор на тренирање (враќам None).
        if len(self.memory) < self.batch_size:
            return None

        batch = self.memory.sample(self.batch_size)
        # stack_states()/to_torch() наместо гол np.array()/torch.tensor() -
        # за kinematics/pixels идентично однесување, за fusion ги стакува/
        # конвертира ДВЕТЕ гранки (kin, img) одделно (agents/networks.py).
        states = to_torch(stack_states(batch.state), self.device)
        actions = torch.tensor(batch.action, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(batch.reward, dtype=torch.float32, device=self.device)
        next_states = to_torch(stack_states(batch.next_state), self.device)
        dones = torch.tensor(batch.done, dtype=torch.float32, device=self.device)

        # Q(s,a) за акциите што реално се земени во batch-от
        q_values = forward_q(self.model, states).gather(1, actions).squeeze(1)

        with torch.no_grad():
            if self.double_dqn:
                # Double DQN трик (од предметот): online мрежата ЈА БИРА
                # најдобрата следна акција, а target мрежата само ЈА
                # ОЦЕНУВА таа акција. Ова го намалува "overestimation bias"
                # - проблемот каде обичниот DQN систематски ги пренагласува
                # Q-вредностите, бидејќи и избира и оценува со истата мрежа.
                next_actions = forward_q(self.model, next_states).argmax(dim=1, keepdim=True)
                next_q = forward_q(self.target_model, next_states).gather(1, next_actions).squeeze(1)
            else:
                next_q = forward_q(self.target_model, next_states).max(dim=1).values
            # Bellman-целта: ако е done, нема идна вредност (множам со 0)
            targets = rewards + self.gamma * next_q * (1 - dones)

        loss = self.criterion(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping - го "ограничувам" градиентот да не биде
        # премногу голем, за да спречам "експлозивни" ажурирања кои го
        # раздвижуваат тренингот (истата причина како за Huber loss погоре
        # - ова директно ми го реши растечкиот loss при тестирање).
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()
        return float(loss.item())

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        # weights_only=True - експлицитно (не се потпирам на library default,
        # кој може да се разликува меѓу torch верзии) - фајлот е обичен
        # state_dict (само тензори), нема причина да се дозволи произволен
        # pickle-код при вчитување, дури и кога фајлот е сопствено-генериран.
        self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        self.update_target_model()
