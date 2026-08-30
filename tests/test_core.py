"""
Мали unit тестови за критичните, лесно-изолирани делови на кодот - НЕ
бараат стартување на целата highway-env симулација (спора/тешка), туку
директно ги тестираат функциите/класите одговорни за конкретни багови кои
ги најдов при ревизијата на кодот (VDN per-agent target masking, replay
buffer capacity/shapes, courtesy shaping, epsilon decay). Ги пишував овие
тестови секогаш кога ќе најдев и поправев некој проблем - за да сум
сигурна дека истиот проблем не се враќа повторно ако случајно нешто
сменам подоцна.

Употреба:
    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import unittest

import numpy as np
import torch

from agents.networks import ReplayBuffer, MultiAgentReplayBuffer, QNetwork
from agents.dqn_agent import DQNAgent
from agents.vdn_agent import VDNAgent
from agents.heuristic_agent import HeuristicAgent
from envs.multi_agent_intersection import MultiAgentIntersectionEnv
from train import compute_epsilon_decay, make_manager
from experiments.parallel import agg_stats


class TestReplayBuffer(unittest.TestCase):
    def test_capacity_wraparound(self):
        # проверувам дека буферот навистина е "circular" - кога е полн,
        # најстарите записи се бришат/презапишуваат, не се чуваат сите засекогаш
        buf = ReplayBuffer(capacity=5)
        for i in range(8):
            buf.push(state=np.array([i]), action=i, reward=float(i), next_state=np.array([i + 1]), done=False)
        # треба да преживеат само последните 5 push-ови (0..7, значи 3,4,5,6,7)
        self.assertEqual(len(buf), 5)
        actions = sorted(t.action for t in buf.buffer)
        self.assertEqual(actions, [3, 4, 5, 6, 7])

    def test_sample_shapes(self):
        buf = ReplayBuffer(capacity=100)
        for i in range(20):
            buf.push(np.array([i, i]), i % 3, float(i), np.array([i + 1, i + 1]), i == 19)
        batch = buf.sample(8)
        self.assertEqual(len(batch.state), 8)
        self.assertEqual(len(batch.action), 8)


class TestMultiAgentReplayBuffer(unittest.TestCase):
    def test_capacity_and_per_agent_shapes(self):
        num_agents = 3
        buf = MultiAgentReplayBuffer(num_agents=num_agents, capacity=10)
        for i in range(15):
            states = [np.array([i, a]) for a in range(num_agents)]
            actions = [a for a in range(num_agents)]
            rewards = [float(i + a) for a in range(num_agents)]
            next_states = [np.array([i + 1, a]) for a in range(num_agents)]
            dones = [False, False, i % 2 == 0]
            buf.push(states, actions, rewards, next_states, dones)
        self.assertEqual(len(buf), 10)  # capacity се почитува (circular overwrite)

        batch = buf.sample(4)
        self.assertEqual(len(batch["states"]), num_agents)
        for i in range(num_agents):
            self.assertEqual(batch["states"][i].shape[0], 4)
            self.assertEqual(batch["dones"][i].shape[0], 4)


class TestEpsilonDecay(unittest.TestCase):
    def test_reaches_min_around_target_fraction(self):
        # проверувам дека epsilon навистина стигнува до epsilon_min околу
        # 70% од тренингот, точно како што треба според дизајнот на функцијата
        episodes = 200
        decay = compute_epsilon_decay(episodes, epsilon_start=1.0, epsilon_min=0.05, target_fraction=0.7)
        eps = 1.0
        for _ in range(int(episodes * 0.7)):
            eps = max(0.05, eps * decay)
        self.assertAlmostEqual(eps, 0.05, delta=1e-6)

    def test_more_episodes_gives_slower_decay(self):
        d_short = compute_epsilon_decay(100)
        d_long = compute_epsilon_decay(1000)
        # повеќе епизоди -> decay поблиску до 1.0, значи epsilon опаѓа побавно
        self.assertGreater(d_long, d_short)


class TestCourtesyPenalty(unittest.TestCase):
    """
    Го тестирам _courtesy_penalty целосно изолирано, БЕЗ да создавам
    вистински highway-env (тоа би било бавно за unit тест) - си
    "измислувам" едноставен env/vehicle објект само со атрибутите што
    методот навистина ги користи.
    """

    class _FakeVehicle:
        def __init__(self, position, speed=0.0):
            self.position = np.array(position, dtype=float)
            self.speed = speed

    class _FakeRoad:
        def __init__(self, vehicles):
            self.vehicles = vehicles

    class _FakeBaseEnv:
        def __init__(self, vehicles, controlled_vehicles):
            self.road = TestCourtesyPenalty._FakeRoad(vehicles)
            self.controlled_vehicles = controlled_vehicles

    def _make_env_instance(self, courtesy_weight, courtesy_distance=12.0, courtesy_speed_threshold=6.0):
        # користам __new__ наместо __init__ за да не мора да создавам вистински gym env
        env = MultiAgentIntersectionEnv.__new__(MultiAgentIntersectionEnv)
        env.courtesy_weight = courtesy_weight
        env.courtesy_distance = courtesy_distance
        env.courtesy_speed_threshold = courtesy_speed_threshold
        return env

    def test_zero_weight_disables_penalty(self):
        env = self._make_env_instance(courtesy_weight=0.0)
        controlled = self._FakeVehicle([0, 0], speed=10.0)
        human = self._FakeVehicle([1, 0], speed=0.0)
        base = self._FakeBaseEnv(vehicles=[controlled, human], controlled_vehicles=[controlled])
        self.assertEqual(env._courtesy_penalty(base, controlled), 0.0)

    def test_slow_vehicle_no_penalty(self):
        env = self._make_env_instance(courtesy_weight=2.0)
        controlled = self._FakeVehicle([0, 0], speed=1.0)  # под threshold=6.0, значи не е "агресивно" возење
        human = self._FakeVehicle([1, 0], speed=0.0)
        base = self._FakeBaseEnv(vehicles=[controlled, human], controlled_vehicles=[controlled])
        self.assertEqual(env._courtesy_penalty(base, controlled), 0.0)

    def test_far_vehicle_no_penalty(self):
        env = self._make_env_instance(courtesy_weight=2.0, courtesy_distance=12.0)
        controlled = self._FakeVehicle([0, 0], speed=10.0)
        human = self._FakeVehicle([50, 0], speed=0.0)  # надвор од courtesy_distance, значи нема казна
        base = self._FakeBaseEnv(vehicles=[controlled, human], controlled_vehicles=[controlled])
        self.assertEqual(env._courtesy_penalty(base, controlled), 0.0)

    def test_close_fast_vehicle_gets_negative_penalty(self):
        env = self._make_env_instance(courtesy_weight=2.0, courtesy_distance=12.0, courtesy_speed_threshold=6.0)
        controlled = self._FakeVehicle([0, 0], speed=10.0)
        human = self._FakeVehicle([6, 0], speed=0.0)  # 6m < 12m, треба да добие казна
        base = self._FakeBaseEnv(vehicles=[controlled, human], controlled_vehicles=[controlled])
        penalty = env._courtesy_penalty(base, controlled)
        self.assertLess(penalty, 0.0)
        # рачна пресметка: closeness = 1 - 6/12 = 0.5 -> penalty = -2.0 * 0.5 = -1.0
        self.assertAlmostEqual(penalty, -1.0, places=6)

    def test_no_human_vehicles_no_penalty(self):
        # MC/DC анализа на "not human_vehicles or speed <= threshold": оваа
        # клауза (нема human возила ВООПШТО) никогаш претходно не беше
        # независно тестирана - во СИТЕ претходни тестови human_vehicles
        # секогаш беше непразна листа. Конкретно, проверливо: мутант кој
        # би го избришал "not human_vehicles or" делот целосно НЕ би бил
        # фатен од ниту еден постоечки тест пред овој.
        env = self._make_env_instance(courtesy_weight=2.0)
        controlled = self._FakeVehicle([0, 0], speed=10.0)  # брзо возење, над threshold
        base = self._FakeBaseEnv(vehicles=[controlled], controlled_vehicles=[controlled])  # нема human возила
        self.assertEqual(env._courtesy_penalty(base, controlled), 0.0)

    def test_boundary_exactly_at_courtesy_distance_no_penalty(self):
        # min_dist ТОЧНО = courtesy_distance=12.0 - кодот користи строго "<",
        # значи точно на границата НЕ смее да добие казна (off-by-one проверка)
        env = self._make_env_instance(courtesy_weight=2.0, courtesy_distance=12.0, courtesy_speed_threshold=6.0)
        controlled = self._FakeVehicle([0, 0], speed=10.0)
        human = self._FakeVehicle([12.0, 0], speed=0.0)
        base = self._FakeBaseEnv(vehicles=[controlled, human], controlled_vehicles=[controlled])
        self.assertEqual(env._courtesy_penalty(base, controlled), 0.0)


class TestHeuristicAgent(unittest.TestCase):
    """
    get_action() е секвенца од 3 бинарни одлуки -> 4 листа (FASTER-никој,
    SLOWER, IDLE, FASTER-далеку). Пред овие тестови, само еден лист беше
    покриен (FASTER-никој) - Node/Edge coverage анализа (CFG со 3 decision
    јазли во низа) бараше уште 3 test requirements за да се допрат
    преостанатите листови, + 2 гранични (danger_distance*0.5 и
    danger_distance се строги "<" прагови во кодот, значи вредност ТОЧНО
    на границата треба да падне на ДРУГАТА страна - класичен off-by-one/
    "<" vs "<=" mutant кој претходно не беше фатен од ниту еден тест).
    """

    @staticmethod
    def _make_obs(other_rel_positions):
        # ред 0 = самото возило (се игнорира во get_action), редовите
        # потоа се "други" возила: presence=1, [rel_x, rel_y] зададени
        n = 1 + len(other_rel_positions)
        obs = np.zeros((n, 7), dtype=np.float32)
        for i, (rx, ry) in enumerate(other_rel_positions, start=1):
            obs[i, 0] = 1.0
            obs[i, 1] = rx
            obs[i, 2] = ry
        return obs.flatten()

    def test_returns_valid_action_indices(self):
        agent = HeuristicAgent()
        obs = np.zeros((6, 7), dtype=np.float32).flatten()  # никој не е присутен -> треба да избере FASTER
        action = agent.get_action(obs)
        self.assertIn(action, (0, 1, 2))
        self.assertEqual(action, HeuristicAgent.ACTIONS["FASTER"])

    def test_very_close_vehicle_triggers_slower(self):
        agent = HeuristicAgent()  # default danger_distance=0.15, половина=0.075
        obs = self._make_obs([(0.05, 0.0)])  # dist=0.05 < 0.075
        self.assertEqual(agent.get_action(obs), HeuristicAgent.ACTIONS["SLOWER"])

    def test_medium_distance_vehicle_triggers_idle(self):
        agent = HeuristicAgent()
        obs = self._make_obs([(0.10, 0.0)])  # 0.075 <= dist=0.10 < 0.15
        self.assertEqual(agent.get_action(obs), HeuristicAgent.ACTIONS["IDLE"])

    def test_far_present_vehicle_still_triggers_faster(self):
        agent = HeuristicAgent()
        obs = self._make_obs([(0.5, 0.0)])  # dist=0.5 >= 0.15 (присутен, но далеку)
        self.assertEqual(agent.get_action(obs), HeuristicAgent.ACTIONS["FASTER"])

    def test_boundary_exactly_at_slower_threshold_falls_to_idle(self):
        # dist ТОЧНО = danger_distance*0.5=0.075 - кодот користи строго "<",
        # значи 0.075 НЕ смее да биде SLOWER (тоа би бил off-by-one багот)
        agent = HeuristicAgent()
        obs = self._make_obs([(0.075, 0.0)])
        self.assertEqual(agent.get_action(obs), HeuristicAgent.ACTIONS["IDLE"])

    def test_boundary_exactly_at_idle_threshold_falls_to_faster(self):
        # dist ТОЧНО = danger_distance=0.15 - исто строго "<", 0.15 НЕ смее да биде IDLE
        agent = HeuristicAgent()
        obs = self._make_obs([(0.15, 0.0)])
        self.assertEqual(agent.get_action(obs), HeuristicAgent.ACTIONS["FASTER"])


class TestDQNAgent(unittest.TestCase):
    def test_train_step_returns_none_until_batch_full_then_float(self):
        torch.manual_seed(0)
        agent = DQNAgent(obs_dim=4, num_actions=3, batch_size=8, buffer_size=100)
        for i in range(7):
            agent.update_memory(np.zeros(4), 0, 1.0, np.zeros(4), False)
        self.assertIsNone(agent.train_step())  # сè уште нема доволно примероци за цел batch (само 7 од 8)
        agent.update_memory(np.zeros(4), 0, 1.0, np.zeros(4), False)
        loss = agent.train_step()  # сега имам точно 8, треба да тргне тренирањето
        self.assertIsInstance(loss, float)
        self.assertFalse(np.isnan(loss))

    def test_get_action_in_range(self):
        agent = DQNAgent(obs_dim=4, num_actions=3)
        action = agent.get_action(np.zeros(4), epsilon=0.0)
        self.assertIn(action, (0, 1, 2))

    def test_double_dqn_produces_different_target_than_vanilla(self):
        """
        Decision coverage за double_dqn=True гранката (train_step()) немаше
        НИЕДЕН тест претходно (IQL секогаш го користи double_dqn=True во
        производство). Само "не пука" би бил слаб oracle - овој тест
        докажува дека гранката РЕАЛНО пресметува поинаку, не само дека се
        извршува без грешка.

        Метод: два агенти (double_dqn=True/False) со БИТ-ИДЕНТИЧНИ
        model/target_model тежини (истиот torch seed) и ИДЕНТИЧЕН batch
        transitions (точно batch_size парчиња - random.sample() врз целосно
        полн buffer секогаш ги враќа СИТЕ елементи, само можеби поинаков
        редослед, што не влијае на mean loss). Единствената разлика меѓу
        нив е double_dqn флагот. За разликата воопшто да значи нешто,
        model МОРА да е различна од target_model (инаку argmax(online) ==
        argmax(target) секогаш, и двете формули случајно би се совпаднале)
        - затоа рачно ја "расипувам" model со трета, независно-seed-ирана
        мрежа, а target_model ја оставам недопрена.
        """
        obs_dim, num_actions, batch_size = 4, 3, 8

        torch.manual_seed(1)
        agent_double = DQNAgent(obs_dim, num_actions, batch_size=batch_size, buffer_size=200, double_dqn=True)
        torch.manual_seed(1)
        agent_vanilla = DQNAgent(obs_dim, num_actions, batch_size=batch_size, buffer_size=200, double_dqn=False)
        # со ист seed=1, model/target_model на двата агенти стартуваат бит-идентични

        torch.manual_seed(99)
        perturbed = QNetwork(obs_dim, num_actions)
        agent_double.model.load_state_dict(perturbed.state_dict())
        agent_vanilla.model.load_state_dict(perturbed.state_dict())

        rng = np.random.default_rng(7)
        transitions = [
            (
                rng.standard_normal(obs_dim).astype(np.float32),
                int(rng.integers(num_actions)),
                float(rng.standard_normal()),
                rng.standard_normal(obs_dim).astype(np.float32),
                False,
            )
            for _ in range(batch_size)
        ]
        for t in transitions:
            agent_double.update_memory(*t)
            agent_vanilla.update_memory(*t)

        loss_double = agent_double.train_step()
        loss_vanilla = agent_vanilla.train_step()

        self.assertIsInstance(loss_double, float)
        self.assertIsInstance(loss_vanilla, float)
        self.assertFalse(np.isnan(loss_double))
        self.assertFalse(np.isnan(loss_vanilla))
        # главната тврдба: double_dqn РЕАЛНО произведува поинаква target-
        # пресметка (не идентичен loss), докажувајќи содржинска разлика, не
        # само дека гранката се извршува
        self.assertNotAlmostEqual(loss_double, loss_vanilla, places=6)


class TestMultiAgentIntersectionEnvIntegration(unittest.TestCase):
    """
    Единствениот тест во целиот пакет што реално создава highway-env
    околина (не "лажирана" како TestCourtesyPenalty погоре) - го
    проверувам договорот (shapes, клучеви, типови) на reset()/step() кој
    ГО ПРЕТПОСТАВУВААТ сите останати делови од проектот (train.py,
    evaluate.py, сите agents/*). Без овој тест, ако highway-env некогаш
    ги промени приватните методи _agent_reward/has_arrived кои ги
    користам (види коментар во requirements.txt), тоа би поминало
    незабележано низ сите останати тестови - тие ниту еднаш не го "лапаат"
    вистинскиот env.
    """

    def test_reset_and_step_contract(self):
        num_agents = 2
        env = MultiAgentIntersectionEnv(num_agents=num_agents)
        try:
            obs = env.reset(seed=0)
            self.assertEqual(len(obs), num_agents)
            for o in obs:
                self.assertEqual(o.shape, (env.obs_dim,))

            actions = [0] * num_agents
            next_obs, rewards, dones, info = env.step(actions)

            self.assertEqual(len(next_obs), num_agents)
            self.assertEqual(len(rewards), num_agents)
            self.assertEqual(len(dones), num_agents)
            for r in rewards:
                self.assertIsInstance(float(r), float)
                self.assertFalse(np.isnan(r))
            self.assertIn("crashed", info)
            self.assertIn("arrived", info)
            self.assertEqual(len(info["crashed"]), num_agents)
            self.assertEqual(len(info["arrived"]), num_agents)
        finally:
            env.close()

    def test_episode_terminates_within_bounded_steps(self):
        # санитетна проверка дека епизодата навистина завршува во разумен
        # број чекори (значи нема бесконечен while-loop во train.py/evaluate.py)
        env = MultiAgentIntersectionEnv(num_agents=2)
        try:
            states = env.reset(seed=1)
            done_all = False
            steps = 0
            max_steps = 200
            rng = np.random.default_rng(0)
            while not done_all and steps < max_steps:
                actions = [int(rng.integers(env.num_actions)) for _ in range(env.num_agents)]
                states, rewards, dones, info = env.step(actions)
                done_all = all(dones)
                steps += 1
            self.assertTrue(done_all, "епизодата не заврши во рамките на max_steps - можен бесконечен loop")
        finally:
            env.close()


class TestLeakedRewardFix(unittest.TestCase):
    """
    Регресионен тест за "истечена" (leaked) награда по пристигнување - види
    class docstring во envs/multi_agent_intersection.py за целосно
    објаснување на багот. highway-env ја продолжува тимската епизода
    додека НЕ завршат СИТЕ контролирани возила, а has_arrived() останува
    True на СЕКОЈ нареден чекор откако возилото еднаш пристигне (лентата е
    права, нема враќање назад) - без поправка, агентот би собирал
    arrived_reward постојано, не само еднаш.

    Со HeuristicAgent (детерминистички, без epsilon-greedy случајност) и
    num_agents=3/seed=1, агент 1 сигурно пристигнува околу чекор 10 додека
    тимските другари сè уште возат до чекор ~20 - потврдено со мануелна
    проверка пред да го напишам овој тест.
    """

    def test_reward_is_zero_after_agent_finishes(self):
        env = MultiAgentIntersectionEnv(num_agents=3)
        agent = HeuristicAgent()
        try:
            states = env.reset(seed=1)
            done_all = False
            step = 0
            arrived_at = None
            rewards_after_arrival = []
            while not done_all and step < 25:
                actions = agent.get_actions(states)
                states, rewards, dones, info = env.step(actions)
                if info["arrived"][1]:
                    if arrived_at is None:
                        arrived_at = step
                        # на самиот чекор на пристигнување, наградата СЀ УШТЕ
                        # треба да е вистинската (не-нула) arrived_reward
                        self.assertGreater(rewards[1], 0.0)
                    else:
                        rewards_after_arrival.append(rewards[1])
                done_all = all(dones)
                step += 1

            self.assertIsNotNone(
                arrived_at,
                "очекував агент 1 да пристигне за овој seed - ако highway-env "
                "однесувањето некогаш се смени, овој тест треба да се ажурира со нов seed",
            )
            self.assertTrue(
                len(rewards_after_arrival) > 0,
                "очекував епизодата да продолжи по пристигнувањето на агент 1 "
                "(тимските другари сè уште возат) - без тоа тестот не го проверува "
                "вистинскиот сценарио на багот",
            )
            self.assertTrue(
                all(r == 0.0 for r in rewards_after_arrival),
                f"агент 1 требаше да добие 0 награда на СИТЕ чекори по пристигнувањето "
                f"(веќе ја добил вистинската награда на чекорот кога завршил), но доби: "
                f"{rewards_after_arrival}",
            )
        finally:
            env.close()


class TestPerAgentRewardsDiffer(unittest.TestCase):
    """
    Го тестира ГЛАВНИОТ придонес на wrapper-от (индивидуална, не
    заедничка/споделена награда по агент) - претходно проверувано само
    посредно (shape/тип/не-NaN во TestMultiAgentIntersectionEnvIntegration),
    никогаш директно дека наградите РЕАЛНО се различни по агент. Мутант кој
    случајно би ја вратил истата споделена награда за сите агенти
    (регресија токму кон проблемот што wrapper-от е дизајниран да го
    реши - види envs/multi_agent_intersection.py) не би бил фатен без ова.

    Наивна верзија (assertNotEqual на прв случаен чекор со идентични акции
    за сите агенти) би била FLAKY - на симетричен почеток со идентична
    акција, наградите МОЖАТ случајно да испаднат многу блиску/еднакви на
    еден чекор, без тоа да значи дека декомпозицијата е расипана. Затоа го
    користам ИСТИОТ детерминистички сценарио како TestLeakedRewardFix
    (seed=1, num_agents=3, агент 1 пристигнува ~чекор 10 додека другите
    возат нормално) - на токму тој чекор, знам СО СИГУРНОСТ дека
    rewards[1] е arrived_reward-базирана вредност додека rewards[0]/[2] се
    "нормални" вожечки награди, значи не-flaky, не случаен тест.
    """

    def test_arrived_agent_reward_differs_from_still_driving_agents(self):
        env = MultiAgentIntersectionEnv(num_agents=3)
        agent = HeuristicAgent()
        try:
            states = env.reset(seed=1)
            done_all = False
            step = 0
            checked = False
            while not done_all and step < 25:
                actions = agent.get_actions(states)
                states, rewards, dones, info = env.step(actions)
                if info["arrived"][1] and not (dones[0] or dones[2]):
                    # чекорот на пристигнување на агент 1, додека 0 и 2 сè уште возат
                    self.assertNotAlmostEqual(rewards[1], rewards[0], places=6)
                    self.assertNotAlmostEqual(rewards[1], rewards[2], places=6)
                    checked = True
                    break
                done_all = all(dones)
                step += 1
            self.assertTrue(
                checked,
                "очекував агент 1 да пристигне додека 0/2 сè уште возат за овој seed - "
                "ако highway-env однесувањето некогаш се смени, тестот треба нов seed",
            )
        finally:
            env.close()


class TestVDNAgentTargetMasking(unittest.TestCase):
    """
    Регресионен тест за баг што го најдов и го поправив во VDN: секој
    агент мора да го маскира САМО својот сопствен bootstrap член
    (rewards_i + gamma*next_q_i*(1-dones_i)), а НЕ целиот Q_tot преку еден
    заеднички "any_done" флаг (види коментар во agents/vdn_agent.py за
    целосно објаснување на багот). Овде само проверувам дека train_step()
    не паѓа и произведува конечен (не-NaN) loss, дури и кога само еден
    агент е "done" во некои примероци од batch-от - точно ситуацијата
    каде багот се манифестираше.
    """

    def test_train_step_stable_with_partial_done_agents(self):
        torch.manual_seed(0)
        num_agents = 3
        agent = VDNAgent(num_agents=num_agents, obs_dim=4, num_actions=3, batch_size=8, buffer_size=200)
        for i in range(20):
            states = [np.random.randn(4).astype(np.float32) for _ in range(num_agents)]
            actions = [np.random.randint(3) for _ in range(num_agents)]
            rewards = [np.random.randn() for _ in range(num_agents)]
            next_states = [np.random.randn(4).astype(np.float32) for _ in range(num_agents)]
            # само еден агент е "done" во секој примерок - точно сценариото што го покриваше багот
            dones = [False] * num_agents
            dones[i % num_agents] = True
            agent.update_memory(states, actions, rewards, next_states, dones)

        loss = agent.train_step()
        self.assertIsInstance(loss, float)
        self.assertFalse(np.isnan(loss))


class TestAggStats(unittest.TestCase):
    """
    experiments/parallel.py::agg_stats е малата helper функција што ја
    користам во СИТЕ 4 experiments/*.py скрипти за да пресметам средина
    ± 95% CI - вреди да е директно тестирана бидејќи секоја погрешна
    бројка таму би се провлекла низ секој агрегиран резултат во проектот.
    """

    def test_mean_and_ci_known_values(self):
        # рачно пресметано: mean([1,2,3,4,5]) = 3, std (population) = sqrt(2) ≈ 1.4142
        # ci95 = 1.96 * std / sqrt(n) = 1.96 * 1.4142 / sqrt(5) ≈ 1.2397
        result = agg_stats([1, 2, 3, 4, 5])
        self.assertAlmostEqual(result["mean"], 3.0, places=6)
        self.assertAlmostEqual(result["ci95"], 1.2397, places=3)

    def test_single_value_has_zero_ci(self):
        # со само 1 вредност нема доверителен интервал - враќам 0.0, не
        # NaN/division-by-zero (n=1 во sqrt(n) е ОК, но n-1=0 некаде другаде би пукнало)
        result = agg_stats([7.0])
        self.assertEqual(result["mean"], 7.0)
        self.assertEqual(result["ci95"], 0.0)

    def test_empty_list_raises_instead_of_silent_nan(self):
        # ISP граница n=0 (веќе имавме n=1, n=5 - n=0 недостасуваше). Одлука
        # при ревизија на кодот: agg_stats([]) СЕГА гласно фрла ValueError
        # наместо тивко да врати {"mean": nan, ...} - agg_stats() секогаш се
        # вика со листа изведена од --seeds, па празна листа секогаш значи
        # погрешно пресметана листа некаде погоре, не валидна ситуација.
        with self.assertRaises(ValueError):
            agg_stats([])


class TestMakeManagerAgentKwargs(unittest.TestCase):
    """
    Регресионен тест за agent_kwargs проследувањето низ
    train.py::make_manager() - потребно за experiments/hyperparameter_study.py
    (варирање на learning_rate/discount_factor). Проверувам дека
    вредностите РЕАЛНО стигнуваат до агентот, не само дека повикот не
    пука (лесно е случајно да се "исфрли" аргумент некаде во синџирот
    make_manager -> VDNAgent.__init__ -> optim.Adam и никој да не забележи).
    """

    def test_vdn_learning_rate_and_gamma_applied(self):
        manager = make_manager("vdn", num_agents=2, obs_dim=4, num_actions=3,
                                agent_kwargs={"learning_rate": 5e-4, "discount_factor": 0.8})
        self.assertIsInstance(manager, VDNAgent)
        self.assertEqual(manager.gamma, 0.8)
        actual_lr = manager.optimizer.param_groups[0]["lr"]
        self.assertAlmostEqual(actual_lr, 5e-4, places=8)

    def test_default_agent_kwargs_none_still_works(self):
        # agent_kwargs=None (default повик, како во train.py CLI употреба
        # без --tag/хиперпараметарски overrides) не смее да падне
        manager = make_manager("iql", num_agents=2, obs_dim=4, num_actions=3)
        self.assertEqual(len(manager.agents), 2)


class TestRunTrainingTag(unittest.TestCase):
    """
    Регресионен тест за tag-базираното именување фајлови во
    train.py::run_training() - клучно за безбедно паралелно стартување
    (experiments/parallel.py). Проверувам дека два running-а со различни
    tag-ови навистина завршуваат со РАЗЛИЧНИ патеки, а не дека случајно
    двата паднат на истото име (тивок бug кој ќе се провлече само при
    вистинско паралелно стартување, тешко да се фати рачно).
    """

    def test_different_tags_produce_different_result_paths(self):
        import uuid

        import train as train_module

        # Уникатен суфикс по ПУШТАЊЕ на тестот (наместо фиксно
        # "tagtestA"/"tagtestB"): ако некое претходно пуштање на овој тест
        # некогаш падне/се прекине НАСРЕД пат (реално ми се случи еднаш -
        # UnicodeEncodeError среде run_training() на не-UTF8 конзола), со
        # фиксни имиња остатоците од таа рунда ќе го "отрујат" СЕКОЕ идно
        # пуштање (history_before веќе би ги содржел тие стари фајлови, па
        # new_files веќе никогаш нема да излезе 2). Со случаен суфикс,
        # секое пуштање е изолирано од сите претходни.
        run_id = uuid.uuid4().hex[:8]
        tags = [f"tagtest{run_id}A", f"tagtest{run_id}B"]

        # addCleanup (не код на крајот на функцијата) гарантира дека
        # чистењето се случува дури и ако некој assertEqual подолу падне -
        # претходната верзија го немаше ова, па токму таквиот пад ги
        # оставаше артефактите зад себе засекогаш.
        def _cleanup():
            for tag in tags:
                for f in train_module.RESULTS_DIR.glob(f"vdn_{tag}*"):
                    f.unlink()

        self.addCleanup(_cleanup)

        for tag in tags:
            train_module.run_training(method="vdn", num_agents=2, episodes=2, seed=0, tag=tag,
                                       checkpoint_every=100)

        history_files = [train_module.RESULTS_DIR / f"vdn_{tag}_history.json" for tag in tags]
        self.assertTrue(all(f.exists() for f in history_files),
                         "очекував по еден history.json за секој tag")


class TestSeedReproducibility(unittest.TestCase):
    """
    Регресионен тест за реален баг најден дури при финалната ревизија на
    целиот проект (по многу сесии работа - откриен само затоа што
    случајно ги споредив два "идентични" повторени тренинзи бајт-по-бајт).
    `train.py::set_global_seed()` го seed-ираше numpy И torch, но НЕ и
    Python-овиот вграден `random` модул - а `agents/networks.py::
    ReplayBuffer`/`MultiAgentReplayBuffer` го користат `random.sample()`
    за batch sampling во ВСЕКОЈ train_step(). Резултат: два стартувања со
    ИСТ --seed сепак произведуваа РАЗЛИЧНИ финални тежини/резултати - env
    динамиката и почетната иницијализација на тежините ВЕЌЕ беа
    репродуцибилни (numpy/torch seed-ирани), но самиот редослед на
    batch-ови при тренирање не беше, бидејќи Python-овиот `random` модул
    си имаше СОПСТВЕНА, OS-ентропија-seed-ирана состојба, целосно
    независна од `np.random.seed()`.
    """

    def test_same_seed_produces_identical_training_history(self):
        import uuid

        import train as train_module

        # Уникатен суфикс - истата причина како во TestRunTrainingTag погоре.
        run_id = uuid.uuid4().hex[:8]
        tags = [f"reprotest{run_id}A", f"reprotest{run_id}B"]

        def _cleanup():
            for tag in tags:
                for f in train_module.RESULTS_DIR.glob(f"vdn_{tag}*"):
                    f.unlink()

        self.addCleanup(_cleanup)

        histories = [
            train_module.run_training(method="vdn", num_agents=2, episodes=8, seed=555, tag=tag,
                                       checkpoint_every=1000, verbose_every=1000)
            for tag in tags
        ]
        self.assertEqual(
            histories[0], histories[1],
            "два run_training() повикувања со ИСТ seed мора да дадат ИДЕНТИЧНА историја "
            "(regression за баг: random.sample() во replay buffer-ите не беше seed-иран)",
        )


class TestEnvClosedOnException(unittest.TestCase):
    """
    Регресионен тест за реален проблем најден при финалната ревизија на
    целиот проект: train.py::run_training() и evaluate.py::evaluate() го
    викаа env.close() БЕЗ try/finally - ако се случеше исклучок среде
    тренирање/евалуација (пр. непознат метод, лош хиперпараметар, NaN
    loss), env-от (и pygame/highway-env ресурсите зад него) никогаш не се
    затвораше. Особено релевантно за pixels/fusion режимите, каде секој
    env носи pygame viewer-и по агент - протекување на многу такви
    ресурси среде долг паралелен тренинг би можело да влијае на
    подоцнежни job-ови во ИСТИОТ worker процес (ProcessPoolExecutor ги
    реупотребува процесите).

    Го тестирам со намерно "расипан" метод (непостоечко име) - ова
    гарантирано фрла ValueError рано, пред env.close() воопшто да се
    достигне на нормалниот пат, значи е чист начин да се провери дека
    finally-блокот РЕАЛНО се извршува.
    """

    def _patch_close(self):
        from envs.multi_agent_intersection import MultiAgentIntersectionEnv

        closed = {"called": False}
        orig_close = MultiAgentIntersectionEnv.close

        def patched_close(env_self):
            closed["called"] = True
            orig_close(env_self)

        MultiAgentIntersectionEnv.close = patched_close
        self.addCleanup(lambda: setattr(MultiAgentIntersectionEnv, "close", orig_close))
        return closed

    def test_run_training_closes_env_even_on_exception(self):
        import train as train_module

        closed = self._patch_close()
        with self.assertRaises(ValueError):
            train_module.run_training(method="not_a_real_method", num_agents=2, episodes=5, seed=0)
        self.assertTrue(closed["called"], "env.close() мора да се повика дури и кога run_training() фрла исклучок")

    def test_evaluate_closes_env_even_on_exception(self):
        import evaluate as evaluate_module

        closed = self._patch_close()
        with self.assertRaises(ValueError):
            evaluate_module.evaluate(method="not_a_real_method", model_prefix=None, num_agents=2,
                                      episodes=2, save=False)
        self.assertTrue(closed["called"], "env.close() мора да се повика дури и кога evaluate() фрла исклучок")


if __name__ == "__main__":
    unittest.main()
