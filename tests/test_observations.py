"""
Тестови за Kinematics/Pixels/Fusion observation режимите (obs_mode) -
проширување надвор од tests/test_core.py (посебен фајл за да не се
"надуе" веќе долгиот test_core.py). Опфаќа:
  - envs/multi_agent_intersection.py: shape/тип на опсервации за pixels/
    fusion, и КРИТИЧНО - дека две различни агентски pixel опсервации во
    ИСТ чекор НЕ се идентични (го наметнува тврдиот услов "своја локална
    слика по агент", не еден ист глобален screenshot за сите).
  - agents/networks.py: build_q_network forward-shape тестови,
    stack_states/to_batch_tensor/forward_q со FusionState.
  - MultiAgentReplayBuffer со fusion states.
  - DQNAgent/VDNAgent/IQLManager train_step() smoke-тестови за pixels/fusion.
  - experiments/robustness.py perturbation функциите.
  - train.py::_apply_modality_dropout / _load_pretrained_img_branch -
    поправка на "modality collapse" кај Fusion (dropout + CNN warm-start).

Употреба:
    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import unittest

import numpy as np
import torch

from agents.networks import (
    FusionState,
    MultiAgentReplayBuffer,
    build_q_network,
    forward_q,
    stack_states,
    to_batch_tensor,
)
from agents.dqn_agent import DQNAgent
from agents.iql_manager import IQLManager
from agents.vdn_agent import VDNAgent
from envs.multi_agent_intersection import MultiAgentIntersectionEnv
from experiments.robustness import add_kinematics_noise, blur_pixels, darken_pixels, drop_branch, make_perturb_fn


class TestEnvPixelObservations(unittest.TestCase):
    """
    Вистински highway-env (не лажиран) - исто оправдување како
    TestMultiAgentIntersectionEnvIntegration во test_core.py: единствениот
    начин реално да се провери дека highway-env-овата
    MultiAgentObservation/GrayscaleObservation интеграција работи како
    очекувам (agent-centric камера по observer_vehicle).
    """

    def test_pixels_mode_shapes_and_per_agent_distinctness(self):
        num_agents = 3
        env = MultiAgentIntersectionEnv(num_agents=num_agents, obs_mode="pixels")
        try:
            obs = env.reset(seed=0)
            self.assertEqual(len(obs), num_agents)
            for o in obs:
                self.assertEqual(o.shape, env.img_shape)
                self.assertEqual(o.dtype, np.uint8)

            # КРИТИЧНО: секој агент гледа СОПСТВЕНА локална слика, не иста
            # глобална screenshot - двете различни агентски слики во ИСТ
            # чекор не смеат да бидат идентични.
            self.assertFalse(np.array_equal(obs[0], obs[1]))
            self.assertFalse(np.array_equal(obs[0], obs[2]))
            self.assertFalse(np.array_equal(obs[1], obs[2]))

            obs2, rewards, dones, info = env.step([0] * num_agents)
            self.assertEqual(len(obs2), num_agents)
            for o in obs2:
                self.assertEqual(o.shape, env.img_shape)
            self.assertFalse(np.array_equal(obs2[0], obs2[1]))
        finally:
            env.close()

    def test_fusion_mode_returns_fusion_state_per_agent(self):
        num_agents = 2
        env = MultiAgentIntersectionEnv(num_agents=num_agents, obs_mode="fusion")
        try:
            obs = env.reset(seed=0)
            self.assertEqual(len(obs), num_agents)
            for o in obs:
                self.assertIsInstance(o, FusionState)
                self.assertEqual(o.kin.shape, (env.obs_dim,))
                self.assertEqual(o.img.shape, env.img_shape)
            self.assertFalse(np.array_equal(obs[0].img, obs[1].img))

            obs2, rewards, dones, info = env.step([0] * num_agents)
            for o in obs2:
                self.assertIsInstance(o, FusionState)
        finally:
            env.close()

    def test_kinematics_mode_unaffected_by_new_code(self):
        # backward-compat регресија: default obs_mode се однесува ИСТО како
        # порано (пред Pixels/Fusion проширувањето).
        env = MultiAgentIntersectionEnv(num_agents=2)
        try:
            self.assertEqual(env.obs_mode, "kinematics")
            self.assertIsNone(env.img_shape)
            obs = env.reset(seed=0)
            for o in obs:
                self.assertIsInstance(o, np.ndarray)
                self.assertEqual(o.shape, (env.obs_dim,))
        finally:
            env.close()

    def test_unknown_obs_mode_raises(self):
        with self.assertRaises(ValueError):
            MultiAgentIntersectionEnv(num_agents=2, obs_mode="audio")


class TestBuildQNetwork(unittest.TestCase):
    def test_pixel_network_forward_shape(self):
        img_shape = (4, 64, 64)
        for dueling in (False, True):
            net = build_q_network("pixels", obs_dim=None, img_shape=img_shape, num_actions=3, dueling=dueling)
            q = net(torch.rand(5, *img_shape) * 255)
            self.assertEqual(q.shape, (5, 3))
            self.assertFalse(torch.isnan(q).any())

    def test_fusion_network_forward_shape(self):
        img_shape = (4, 64, 64)
        for dueling in (False, True):
            net = build_q_network("fusion", obs_dim=10, img_shape=img_shape, num_actions=3, dueling=dueling)
            q = net(torch.rand(5, 10), torch.rand(5, *img_shape) * 255)
            self.assertEqual(q.shape, (5, 3))
            self.assertFalse(torch.isnan(q).any())

    def test_kinematics_network_unchanged(self):
        # backward-compat: build_q_network("kinematics", ...) враќа истите
        # класи (QNetwork/DuelingQNetwork) со истиот интерфејс како порано.
        net = build_q_network("kinematics", obs_dim=10, img_shape=None, num_actions=3, dueling=False)
        q = net(torch.rand(5, 10))
        self.assertEqual(q.shape, (5, 3))

    def test_unknown_obs_mode_raises(self):
        with self.assertRaises(ValueError):
            build_q_network("audio", obs_dim=10, img_shape=None, num_actions=3, dueling=False)


class TestObsHelpers(unittest.TestCase):
    def test_stack_states_plain_arrays(self):
        stacked = stack_states([np.array([1.0, 2.0]), np.array([3.0, 4.0])])
        self.assertEqual(stacked.shape, (2, 2))

    def test_stack_states_fusion(self):
        states = [
            FusionState(kin=np.zeros(4), img=np.ones((2, 3, 3))),
            FusionState(kin=np.ones(4), img=np.zeros((2, 3, 3))),
        ]
        stacked = stack_states(states)
        self.assertIsInstance(stacked, FusionState)
        self.assertEqual(stacked.kin.shape, (2, 4))
        self.assertEqual(stacked.img.shape, (2, 2, 3, 3))

    def test_to_batch_tensor_fusion_adds_batch_dim(self):
        state = FusionState(kin=np.zeros(4), img=np.ones((2, 3, 3)))
        t = to_batch_tensor(state, device="cpu")
        self.assertIsInstance(t, FusionState)
        self.assertEqual(tuple(t.kin.shape), (1, 4))
        self.assertEqual(tuple(t.img.shape), (1, 2, 3, 3))

    def test_forward_q_dispatches_fusion_vs_plain(self):
        # img_shape мора да е доволно голем за 3-те stride-2 конволутивни
        # слоја во _CNNEncoder (kernel 5/3/3) реално да произведат валиден
        # излез - 32x32 е безбеден минимум (64x64, реалната production
        # големина, е далеку над ова).
        fusion_net = build_q_network("fusion", obs_dim=4, img_shape=(2, 32, 32), num_actions=3, dueling=False)
        q = forward_q(fusion_net, FusionState(kin=torch.rand(1, 4), img=torch.rand(1, 2, 32, 32) * 255))
        self.assertEqual(q.shape, (1, 3))

        plain_net = build_q_network("kinematics", obs_dim=4, img_shape=None, num_actions=3, dueling=False)
        q2 = forward_q(plain_net, torch.rand(1, 4))
        self.assertEqual(q2.shape, (1, 3))


class TestMultiAgentReplayBufferFusion(unittest.TestCase):
    def test_sample_stacks_fusion_states_per_agent(self):
        num_agents = 2
        buf = MultiAgentReplayBuffer(num_agents=num_agents, capacity=20)
        for i in range(10):
            states = [FusionState(kin=np.full(4, i), img=np.full((2, 3, 3), i)) for _ in range(num_agents)]
            next_states = [
                FusionState(kin=np.full(4, i + 1), img=np.full((2, 3, 3), i + 1)) for _ in range(num_agents)
            ]
            buf.push(states, [0, 1], [1.0, 2.0], next_states, [False, False])

        batch = buf.sample(4)
        for i in range(num_agents):
            self.assertIsInstance(batch["states"][i], FusionState)
            self.assertEqual(batch["states"][i].kin.shape, (4, 4))
            self.assertEqual(batch["states"][i].img.shape, (4, 2, 3, 3))


class TestAgentsTrainStepPixelsFusion(unittest.TestCase):
    """
    Смоук-тестови (без вистински highway-env, обични random state-ови) дека
    DQNAgent/VDNAgent train_step() работи и за pixels/fusion режим - истиот
    дух како TestVDNAgentTargetMasking во test_core.py.
    """

    @staticmethod
    def _random_state(obs_mode, obs_dim, img_shape):
        if obs_mode == "pixels":
            return (np.random.rand(*img_shape) * 255).astype(np.uint8)
        return FusionState(
            kin=np.random.randn(obs_dim).astype(np.float32),
            img=(np.random.rand(*img_shape) * 255).astype(np.uint8),
        )

    def test_dqn_agent_pixels_train_step(self):
        torch.manual_seed(0)
        img_shape = (2, 32, 32)  # доволно голем за 3 stride-2 конволутивни слоја
        agent = DQNAgent(obs_dim=None, num_actions=3, batch_size=4, buffer_size=50,
                          obs_mode="pixels", img_shape=img_shape)
        for _ in range(6):
            s = self._random_state("pixels", None, img_shape)
            ns = self._random_state("pixels", None, img_shape)
            agent.update_memory(s, np.random.randint(3), float(np.random.randn()), ns, False)
        loss = agent.train_step()
        self.assertIsInstance(loss, float)
        self.assertFalse(np.isnan(loss))
        action = agent.get_action(self._random_state("pixels", None, img_shape), epsilon=0.0)
        self.assertIn(action, (0, 1, 2))

    def test_dqn_agent_fusion_train_step(self):
        torch.manual_seed(0)
        img_shape = (2, 32, 32)  # доволно голем за 3 stride-2 конволутивни слоја
        agent = DQNAgent(obs_dim=6, num_actions=3, batch_size=4, buffer_size=50,
                          obs_mode="fusion", img_shape=img_shape)
        for _ in range(6):
            s = self._random_state("fusion", 6, img_shape)
            ns = self._random_state("fusion", 6, img_shape)
            agent.update_memory(s, np.random.randint(3), float(np.random.randn()), ns, False)
        loss = agent.train_step()
        self.assertIsInstance(loss, float)
        self.assertFalse(np.isnan(loss))

    def test_vdn_agent_fusion_train_step(self):
        torch.manual_seed(0)
        num_agents, img_shape = 2, (2, 32, 32)  # доволно голем за 3 stride-2 конволутивни слоја
        agent = VDNAgent(num_agents=num_agents, obs_dim=6, num_actions=3, batch_size=4, buffer_size=50,
                          obs_mode="fusion", img_shape=img_shape)
        for _ in range(6):
            states = [self._random_state("fusion", 6, img_shape) for _ in range(num_agents)]
            next_states = [self._random_state("fusion", 6, img_shape) for _ in range(num_agents)]
            agent.update_memory(
                states,
                [np.random.randint(3) for _ in range(num_agents)],
                [float(np.random.randn()) for _ in range(num_agents)],
                next_states,
                [False] * num_agents,
            )
        loss = agent.train_step()
        self.assertIsInstance(loss, float)
        self.assertFalse(np.isnan(loss))
        actions = agent.get_actions([self._random_state("fusion", 6, img_shape) for _ in range(num_agents)],
                                     epsilon=0.0)
        self.assertEqual(len(actions), num_agents)

    def test_iql_manager_forwards_obs_mode_to_each_agent(self):
        # регресија за obs_mode/img_shape kwargs "минуваат низ" IQLManager
        # (обично **dqn_kwargs, iql_manager.py воопшто не е менет) до секој
        # внатрешен DQNAgent.
        img_shape = (2, 32, 32)  # доволно голем за 3 stride-2 конволутивни слоја
        manager = IQLManager(2, None, 3, double_dqn=True, obs_mode="pixels", img_shape=img_shape)
        self.assertEqual(len(manager.agents), 2)
        for agent in manager.agents:
            self.assertEqual(agent.obs_mode, "pixels")


class TestRobustnessPerturbations(unittest.TestCase):
    def test_kinematics_noise_changes_values_preserves_shape_and_presence(self):
        np.random.seed(0)
        kin = np.zeros(14, dtype=np.float32)  # 2 возила x 7 [presence,x,y,vx,vy,cos_h,sin_h]
        kin[0] = 1.0  # presence на возило 0
        kin[7] = 1.0  # presence на возило 1
        noisy = add_kinematics_noise(kin, sigma=0.2)
        self.assertEqual(noisy.shape, kin.shape)
        self.assertEqual(noisy[0], 1.0)  # presence-бит недопрен
        self.assertEqual(noisy[7], 1.0)
        self.assertFalse(np.array_equal(noisy, kin))  # некоја континуирана вредност се променила
        self.assertTrue(np.all(noisy <= 1.0) and np.all(noisy >= -1.0))  # clip во нормализираниот опсег

    def test_zero_sigma_noise_is_identity(self):
        kin = np.random.uniform(-1, 1, size=14).astype(np.float32)
        noisy = add_kinematics_noise(kin, sigma=0.0)
        np.testing.assert_allclose(noisy, kin, atol=1e-6)

    def test_blur_reduces_variance(self):
        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, size=(2, 32, 32), dtype=np.uint8)
        blurred = blur_pixels(img, sigma=2.0)
        self.assertEqual(blurred.shape, img.shape)
        self.assertEqual(blurred.dtype, np.uint8)
        self.assertLess(blurred.astype(float).var(), img.astype(float).var())

    def test_darken_reduces_mean_without_changing_shape(self):
        img = np.full((2, 8, 8), 200, dtype=np.uint8)
        darkened = darken_pixels(img, factor=0.3)
        self.assertEqual(darkened.shape, img.shape)
        self.assertAlmostEqual(darkened.mean(), 60.0, delta=1.0)

    def test_darken_factor_one_is_identity(self):
        img = np.full((2, 8, 8), 123, dtype=np.uint8)
        self.assertTrue(np.array_equal(darken_pixels(img, factor=1.0), img))

    def test_drop_branch_zeros_exactly_one_branch(self):
        state = FusionState(kin=np.ones(4), img=np.full((2, 3, 3), 7, dtype=np.uint8))

        dropped_kin = drop_branch(state, "kin")
        self.assertTrue(np.array_equal(dropped_kin.kin, np.zeros(4)))
        self.assertTrue(np.array_equal(dropped_kin.img, state.img))  # img недопрена

        dropped_img = drop_branch(state, "img")
        self.assertTrue(np.array_equal(dropped_img.img, np.zeros_like(state.img)))
        self.assertTrue(np.array_equal(dropped_img.kin, state.kin))  # kin недопрена

    def test_drop_branch_invalid_which_raises(self):
        with self.assertRaises(ValueError):
            drop_branch(FusionState(kin=np.ones(4), img=np.zeros((2, 3, 3))), "audio")

    def test_make_perturb_fn_applies_to_every_agent(self):
        perturb = make_perturb_fn("pixel_dark", 0.5)
        out = perturb([np.full((2, 4, 4), 100, dtype=np.uint8) for _ in range(3)])
        self.assertEqual(len(out), 3)
        for o in out:
            self.assertAlmostEqual(o.mean(), 50.0, delta=1.0)

    def test_make_perturb_fn_drop_kin_on_fusion_states(self):
        perturb = make_perturb_fn("drop_kin", strength=None)
        out = perturb([FusionState(kin=np.ones(4), img=np.ones((2, 3, 3))) for _ in range(2)])
        for o in out:
            self.assertTrue(np.array_equal(o.kin, np.zeros(4)))
            self.assertTrue(np.array_equal(o.img, np.ones((2, 3, 3))))

    def test_make_perturb_fn_unknown_kind_raises(self):
        perturb = make_perturb_fn("audio", 1.0)
        with self.assertRaises(ValueError):
            perturb([np.zeros(4)])


class TestModalityDropout(unittest.TestCase):
    """
    Тестови за train.py::_apply_modality_dropout - "поправката" на
    modality collapse кај Fusion (повремено гасење на kin-гранката за
    време на тренирањето, види коментар кај agents/networks.py::drop_branch).
    Функционално веќе проверена (smoke-тестови + целосен тренинг во
    experiments/fusion_fix_study.py), но немаше брз unit тест до финалната
    ревизија на проектот.
    """

    def test_zero_prob_is_complete_noop(self):
        from train import _apply_modality_dropout

        states = [FusionState(kin=np.array([1.0, 2.0]), img=np.zeros((2, 3, 3))) for _ in range(3)]
        out = _apply_modality_dropout(states, 0.0)
        self.assertIs(out, states)  # рано-враќање - истиот објект, не дури ни копија

    def test_prob_one_always_zeros_kin_but_not_img(self):
        from train import _apply_modality_dropout

        states = [FusionState(kin=np.array([1.0, 2.0]), img=np.full((2, 3, 3), 7.0)) for _ in range(5)]
        out = _apply_modality_dropout(states, 1.0)
        for o in out:
            self.assertTrue(np.array_equal(o.kin, np.zeros(2)))
            self.assertTrue(np.array_equal(o.img, np.full((2, 3, 3), 7.0)))

    def test_ignores_non_fusion_states_regardless_of_prob(self):
        # kinematics/pixels states (обични numpy array-и, не FusionState) -
        # dropout-от НЕ треба да прави ништо со нив, дури и со prob=1.0.
        from train import _apply_modality_dropout

        states = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        out = _apply_modality_dropout(states, 1.0)
        for s, o in zip(states, out):
            self.assertTrue(np.array_equal(s, o))


class TestPretrainedImgBranchWarmStart(unittest.TestCase):
    """Тестови за train.py::_load_pretrained_img_branch (CNN warm-start за Fusion)."""

    def test_loads_matching_encoder_weights_into_fusion_img_branch(self):
        import uuid

        from train import RESULTS_DIR, _load_pretrained_img_branch

        img_shape = (2, 32, 32)  # доволно голем за 3-те stride-2 конволутивни слоја
        run_id = uuid.uuid4().hex[:8]
        prefix = str(RESULTS_DIR / f"unittest_pixels_{run_id}")
        self.addCleanup(lambda: [f.unlink() for f in RESULTS_DIR.glob(f"unittest_pixels_{run_id}*")])

        # "претренирана" Pixels-only мрежа - само ги зачувувам случајните
        # (не вистински тренирани) тежини, доволно за да проверам дека
        # state_dict-от РЕАЛНО се пренесува во Fusion img_branch-от, не
        # дека самите тежини се "точни"/корисни.
        pixels_agent = VDNAgent(num_agents=2, obs_dim=None, num_actions=3, dueling=True,
                                 obs_mode="pixels", img_shape=img_shape)
        pixels_agent.save(prefix)

        fusion_manager = VDNAgent(num_agents=2, obs_dim=4, num_actions=3, dueling=True,
                                   obs_mode="fusion", img_shape=img_shape)
        _load_pretrained_img_branch(fusion_manager, "vdn", prefix, fusion_manager.device)

        for i in range(2):
            pretrained_sd = pixels_agent.models[i].encoder.state_dict()
            for key, val in pretrained_sd.items():
                self.assertTrue(
                    torch.equal(val, fusion_manager.models[i].img_branch.state_dict()[key]),
                    f"agent{i} img_branch.{key} не се совпаѓа со претренираните тежини",
                )
                # target мрежата исто мора да ги "види" претренираните тежини
                # (manager.update_target_model() се повикува ВНАТРЕ во _load_pretrained_img_branch)
                self.assertTrue(
                    torch.equal(val, fusion_manager.target_models[i].img_branch.state_dict()[key]),
                    f"agent{i} target_models img_branch.{key} не е синхронизирана",
                )

    def test_non_pixels_source_raises_clear_error(self):
        import uuid

        from train import RESULTS_DIR, _load_pretrained_img_branch

        run_id = uuid.uuid4().hex[:8]
        prefix = str(RESULTS_DIR / f"unittest_notpixels_{run_id}")
        self.addCleanup(lambda: [f.unlink() for f in RESULTS_DIR.glob(f"unittest_notpixels_{run_id}*")])

        # kinematics модел (нема "encoder."-префиксирани клучеви во state_dict-от)
        # - треба ЈАСНО да падне, не тивко "да успее" без реално да пренесе ништо.
        kin_agent = VDNAgent(num_agents=1, obs_dim=4, num_actions=3, dueling=True)
        kin_agent.save(prefix)

        fusion_manager = VDNAgent(num_agents=1, obs_dim=4, num_actions=3, dueling=True,
                                   obs_mode="fusion", img_shape=(2, 32, 32))
        with self.assertRaises(ValueError):
            _load_pretrained_img_branch(fusion_manager, "vdn", prefix, fusion_manager.device)


if __name__ == "__main__":
    unittest.main()
