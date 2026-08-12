"""
Independent Q-Learning (IQL): секој агент е одделен DQNAgent
(agents/dqn_agent.py) кој учи целосно НЕЗАВИСНО, третирајќи ги другите
возила само како дел од (нестационарната) околина - значи агентот 1
воопшто не знае дека агент 2 постои како "агент", само го гледа во
опсервацијата како уште едно возило на патот.

Ова е намерно наједноставниот мулти-агентен baseline во проектот, и
служи за споредба со VDN/CTDE пристапот. Директно го тестира тврдењето од
белешките за MARL: "Од гледна точка на еден агент, средината станува
понестабилна: другите агенти учат и ја менуваат нивната политика." Со
други зборови - додека агент 1 се обидува да научи добра политика,
"подот му се движи под нозете", бидејќи и агент 2 и агент 3 во исто
време ги менуваат СВОИТЕ политики. Ова е клучната разлика со VDN, каде
тренирањето е централизирано и агентите учат заедно.
"""
from __future__ import annotations

import numpy as np

from agents.dqn_agent import DQNAgent


class IQLManager:
    """Само "менаџира" листа од независни DQNAgent-и, по еден за секое возило."""

    def __init__(self, num_agents: int, obs_dim: int, num_actions: int, **dqn_kwargs):
        self.num_agents = num_agents
        self.agents = [DQNAgent(obs_dim, num_actions, **dqn_kwargs) for _ in range(num_agents)]

    def get_actions(self, states: list[np.ndarray], epsilon: float) -> list[int]:
        # секој агент одлучува целосно самостојно, врз основа само на својата опсервација
        return [agent.get_action(s, epsilon) for agent, s in zip(self.agents, states)]

    def update_memory(self, states, actions, rewards, next_states, dones):
        # секој агент ги паметам своите искуства во СВОЈ посебен replay buffer
        for i, agent in enumerate(self.agents):
            agent.update_memory(states[i], actions[i], rewards[i], next_states[i], dones[i])

    def train_step(self) -> list[float | None]:
        # секој агент се тренира ЗАСЕБНО - нема заедничка loss функција како кај VDN
        return [agent.train_step() for agent in self.agents]

    def update_target_model(self):
        for agent in self.agents:
            agent.update_target_model()

    def save(self, path_prefix: str):
        for i, agent in enumerate(self.agents):
            agent.save(f"{path_prefix}_agent{i}.pt")

    def load(self, path_prefix: str):
        for i, agent in enumerate(self.agents):
            agent.load(f"{path_prefix}_agent{i}.pt")
