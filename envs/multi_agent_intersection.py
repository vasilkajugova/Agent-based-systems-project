"""
Ова е мојата сопствена "обвивка" (wrapper) околу околината intersection-v2
од библиотеката highway-env. Ми требаше затоа што highway-env "из кутија"
не поддржува сè што ми е потребно за мулти-агентен RL проект:

  1. Можност сама да одберам колку возила да контролирам (колку "агенти"
     да учествуваат во учењето).
  2. Секој агент да добива СВОЈА, посебна награда. highway-env стандардно
     враќа само ЕДНА заедничка (просечна) награда за сите возила заедно
     преку env.step() - а мене ми треба секој агент да знае "колку добро
     се однесував ЈАС", инаку IQL (Independent Q-Learning) воопшто не би
     можел да учи, бидејќи секој агент учи од сопствената награда.
  3. Опсервацијата (она "што го гледа" агентот во секој момент) да биде
     еден обичен вектор броеви, а не матрица - невронските мрежи што ги
     користам во проектот (обични fully-connected MLP мрежи) очекуваат
     влез во облик на вектор, затоа опсервацијата ја "спрамувам" (flatten)
     во еден ред.
  4. Ист, едноставен интерфејс и за IQL и за VDN агентите, за да не морам
     да пишувам различен код за секој метод одделно.

Дел за кој потрошив доста време додека сфатив зошто не работи: highway-env
внатрешно има функција `_agent_reward(action, vehicle)` која ја пресметува
наградата за секое возило поединечно, но таа НЕ е достапна преку
стандардниот gymnasium `step()` повик кога се користи MultiAgentAction/
MultiAgentObservation (токму она што ми треба мене за мулти-агентен
сетинг). Затоа морам да "влезам подлабоко" преку `env.unwrapped` и да ја
викам таа функција директно, за секое возило одделно - тоа е всушност
најважниот дел од овој фајл (методот `step()` подолу).
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
import highway_env  # noqa: F401  (само со тоа што се импортира, highway-env си ги регистрира своите околини кај gymnasium)

gym.register_envs(highway_env)


# Тука ги имам сите поставки на околината на едно место, за полесно да ги
# менувам при експерименти (пр. за ablation студиите подоцна во проектот).
DEFAULT_CONFIG = {
    "controlled_vehicles": 3,      # колку возила контролира RL агентот(ите)
    "initial_vehicle_count": 8,    # вкупно возила на почеток (вклучувајќи ги контролираните)
    "spawn_probability": 0.6,      # веројатност да "влезе" ново (human-driven) возило секој чекор
    "duration": 20,                # секунди по епизода - НАМЕРНО кратко, за побрз тренинг/тестирање
    "destination": "o1",
    "collision_reward": -5,        # казна кога ќе се судри возилото
    "high_speed_reward": 1,        # награда за возење со поголема брзина
    "arrived_reward": 2,           # награда кога возилото ќе стигне до дестинацијата
    "reward_speed_range": [7.0, 9.0],
    "observation": {
        "type": "MultiAgentObservation",
        "observation_config": {
            "type": "Kinematics",
            "vehicles_count": 6,   # колку најблиски возила "гледа" секој агент (себеси + 5 други)
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            "features_range": {
                "x": [-100, 100],
                "y": [-100, 100],
                "vx": [-15, 15],
                "vy": [-15, 15],
            },
            "absolute": False,     # координатите се РЕЛАТИВНИ спрема самото возило, не апсолутни на патот
            "flatten": True,
            "normalize": True,     # сите вредности се "спакувани" во опсег [-1, 1] - подобро за учење
        },
    },
    "action": {
        "type": "MultiAgentAction",
        "action_config": {
            "type": "DiscreteMetaAction",
            "longitudinal": True,      # дозволени се акции забавување/забрзување
            "lateral": False,          # смена на лента е ИСКЛУЧЕНА (раскрсницата има фиксни патеки)
            "target_speeds": [0, 4.5, 9],
        },
    },
}


class MultiAgentIntersectionEnv:
    """
    Едноставна обвивка со фиксен број дискретни акции и посебна награда
    по агент.

    Начин на употреба (намерно личи на PettingZoo "parallel" API кое го
    работевме/спомнавме на предметот, само поедноставено за овој проект):

        obs_list = env.reset()                             -> листа од np.ndarray, должина = num_agents
        obs_list, rewards, dones, info = env.step(actions)  -> actions: листа од int (по еден по агент)

    Мојот оригинален придонес тука - "courtesy" (фер/љубезно однесување)
    reward shaping:
        highway-env стандардно наградува само брзина, безбедност и
        пристигнување, без разлика како контролираното возило влијае на
        останатите, human-driven (неконтролирани) возила во близина.
        Затоа додадов дополнителен член во наградата кој КАЗНУВА агресивно
        (брзо) возење кога агентот е близу до human-driven возило - идеата
        е "не сечи го патот на другите само за да заштедиш малку награда
        за себе". Ова е целосно конфигурабилно (courtesy_weight=0 го
        исклучува целосно), за да можам директно да споредам со и без него.

    Втора техничка ревизија - поправка на "истечена" (leaked) награда по
    пристигнување:
        highway-env ја прекинува ЦЕЛАТА тимска епизода дури кога СИТЕ
        контролирани возила ќе пристигнат (или штом БИЛО КОЕ ќе се судри) -
        види `_is_terminated()` во highway_env/envs/intersection_env.py.
        Значи ако агент 1 пристигне на чекор 5, а агент 2 сè уште вози до
        чекор 15, тимската епизода трае до чекор 15. Проблемот: `has_arrived`
        не е еднократен настан туку проверка на ТЕКОВНА позиција - штом
        возилото еднаш ја помине точката на пристигнување, си останува
        "arrived" на секој нареден чекор (лентата е права, нема свртување
        назад), па `_agent_reward()` враќа arrived_reward (+2) на СЕКОЈ од
        тие преостанати 10 чекори, не само еднаш! Го проверив ова емпириски
        со реалните истренирани модели - кај VDN, околу 30% од целата
        собрана награда во евалуацијата доаѓаше токму од овој ефект (агент
        стои и "чека" тимските другари да завршат, а во меѓувреме бесплатно
        собира награда секој чекор) - и тоа асиметрично помеѓу методите
        (VDN ~29%, IQL ~15%), значи директно го нарушуваше главниот наод
        "VDN има повисока просечна награда од IQL". Поправката: паметам
        `self._active` (кој агент сè уште НЕ завршил) и штом еднаш агент
        заврши, неговата награда на СИТЕ натамошни чекори е 0 - неговата
        вистинска (не-нула) награда веќе ја добил токму на чекорот кога
        завршил.
    """

    def __init__(
        self,
        num_agents: int = 3,
        config_overrides: dict | None = None,
        render: bool = False,
        render_mode: str | None = None,
        courtesy_weight: float = 0.0,
        courtesy_distance: float = 12.0,
        courtesy_speed_threshold: float = 6.0,
    ):
        cfg = {**DEFAULT_CONFIG, "controlled_vehicles": num_agents}
        if config_overrides:
            cfg.update(config_overrides)
        self.num_agents = num_agents
        self.courtesy_weight = courtesy_weight
        self.courtesy_distance = courtesy_distance
        self.courtesy_speed_threshold = courtesy_speed_threshold

        # render_mode="rgb_array" ми е потребен за watch.py (снимање
        # фрејмови за GIF/споредба), а не влијае на постоечкото однесување:
        # ако не се зададе експлицитно, се однесува точно како порано (render=True -> "human").
        if render_mode is None:
            render_mode = "human" if render else None
        self.env = gym.make("intersection-v2", config=cfg, render_mode=render_mode)
        self.env.reset()  # мора да се повика еднаш на почеток за да се "постават" action_space/observation_space

        # сите агенти делат ист (Discrete) простор на акции, затоа земам од првиот
        self.num_actions = self.env.action_space[0].n
        # Забелешка: MultiAgentObservation во highway-env не го проследува
        # "flatten" флагот правилно до внатрешните KinematicObservation
        # објекти (останува во облик (vehicles_count, 7) наместо еден
        # вектор), па го правам flatten-ирањето рачно во методот _flatten.
        raw_shape = self.env.observation_space[0].shape
        self.obs_dim = int(np.prod(raw_shape))

    def _flatten(self, obs_tuple):
        # секоја опсервација (matrix vehicles_count x 7) ја "распластувам" во еден вектор
        return [np.asarray(o, dtype=np.float32).flatten() for o in obs_tuple]

    def _courtesy_penalty(self, base_env, controlled_vehicle) -> float:
        """
        Го пресметува мојот "courtesy" (фер однесување) shaping член.

        Идеата е едноставна: ако возилото вози брзо (над courtesy_speed_threshold)
        додека е блиску (под courtesy_distance) до кое било human-driven
        (неконтролирано) возило, го казнувам за тоа - колку е поблиску,
        толку поголема казна. Ако courtesy_weight=0 (default вредност),
        функцијава секогаш враќа 0 и воопшто не влијае на резултатите -
        така можам чисто да споредам "со" и "без" овој дел од наградата.
        """
        if self.courtesy_weight == 0.0:
            return 0.0

        human_vehicles = [
            v for v in base_env.road.vehicles if v not in base_env.controlled_vehicles
        ]
        if not human_vehicles or controlled_vehicle.speed <= self.courtesy_speed_threshold:
            return 0.0

        min_dist = min(
            np.linalg.norm(controlled_vehicle.position - hv.position) for hv in human_vehicles
        )
        if min_dist < self.courtesy_distance:
            # колку е поблиску, толку е поголема казната (линеарно опаѓа со растојанието)
            closeness = 1.0 - (min_dist / self.courtesy_distance)
            return -self.courtesy_weight * closeness
        return 0.0

    def reset(self, seed: int | None = None):
        obs, _info = self.env.reset(seed=seed)
        # "active" го паметам за да можам да ја "исечам" наградата за секој
        # агент штом еднаш заврши (се судри/пристигне) - видете коментарот
        # во step() подолу зошто ми е ова потребно (поправка на "истечена"
        # награда по пристигнување).
        self._active = [True] * self.num_agents
        return self._flatten(obs)

    def step(self, actions: list[int]):
        obs, _shared_reward, terminated, truncated, info = self.env.step(tuple(actions))

        # Тука е клучниот дел: ги извлекувам индивидуалните награди по
        # агент директно преку внатрешната highway-env функција
        # _agent_reward(action, vehicle), бидејќи стандардниот env.step()
        # враќа само една заедничка (_shared_reward) награда за сите.
        base_env = self.env.unwrapped
        per_agent_rewards = [
            base_env._agent_reward(a, v) + self._courtesy_penalty(base_env, v)
            for a, v in zip(actions, base_env.controlled_vehicles)
        ]
        per_agent_crashed = [v.crashed for v in base_env.controlled_vehicles]
        per_agent_arrived = [base_env.has_arrived(v) for v in base_env.controlled_vehicles]

        done = bool(terminated or truncated)
        # секој агент е "done" ако целата епизода заврши, ИЛИ ако токму тој се судрил/стигнал
        dones = [done or c or a for c, a in zip(per_agent_crashed, per_agent_arrived)]

        # Поправка на "истечена" награда (види class docstring погоре за
        # целосно објаснување): агент кој ВЕЌЕ бил done на некој ПРЕТХОДЕН
        # чекор (self._active[i] == False пред овој чекор) добива 0 награда
        # тука - неговата вистинска награда (пр. arrived_reward) веќе му е
        # доделена точно на чекорот кога завршил, додека self._active[i]
        # сè уште беше True. Без ова, has_arrived() останува True на секој
        # нареден чекор (лентата е права, нема враќање назад), па агентот
        # би собирал arrived_reward постојано додека ги чека тимските
        # другари да завршат - систематски ги "надувува" собраните награди
        # (и во тренинг, и во евалуација), различно по метод/епизода во
        # зависност од тоа колку долго траела епизодата ПОТОА.
        per_agent_rewards = [
            r if was_active else 0.0
            for r, was_active in zip(per_agent_rewards, self._active)
        ]
        self._active = [a and not d for a, d in zip(self._active, dones)]

        info_out = {
            "crashed": per_agent_crashed,
            "arrived": per_agent_arrived,
            "shared_reward": _shared_reward,  # ја чувам и оригиналната заедничка награда, само за увид/debug
        }
        return self._flatten(obs), per_agent_rewards, dones, info_out

    def close(self):
        self.env.close()


if __name__ == "__main__":
    # Брз "smoke test" - само неколку случајни чекори за да проверам дека
    # обвивката воопшто работи, пред да почнам да пишувам код за тренинг.
    # Секогаш кога ќе го менувам овој фајл, прво го пуштам ова.
    env = MultiAgentIntersectionEnv(num_agents=3)
    obs = env.reset(seed=0)
    print(f"[OK] num_agents={env.num_agents}  obs_dim={env.obs_dim}  num_actions={env.num_actions}")
    print(f"[OK] obs[0].shape = {obs[0].shape}")
    total_r = [0.0] * env.num_agents
    for step in range(15):
        actions = [np.random.randint(env.num_actions) for _ in range(env.num_agents)]
        obs, rewards, dones, info = env.step(actions)
        total_r = [t + r for t, r in zip(total_r, rewards)]
        if all(dones):
            break
    print(f"[OK] завршено по {step + 1} чекори, кумулативни награди: {total_r}")
    env.close()
