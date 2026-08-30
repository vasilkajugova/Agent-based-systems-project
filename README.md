*[Read in English → README.en.md](README.en.md)*

# Мулти-агентно учење со поттикнување за координација на автономни возила на раскрсница

Проект од предметот **Агентно-базирани системи**
(MDP → Bellman/DP → RL основи → Deep RL/DQN → Multi-Agent RL).

## Истражувачко прашање

Дали агенти тренирани со **CTDE** (Centralized Training, Decentralized
Execution, конкретно **VDN - Value Decomposition Networks**) постигнуваат
подобар компромис помеѓу безбедност (стапка на судири) и ефикасност
(просечни чекори до дестинација / arrival rate) во споредба со
**Independent Q-Learning (IQL)** и **rule-based baseline**, во сценарио
со раскрсница со мешан сообраќај (контролирани + human-driven возила)?

## Мојот придонес

1. **Сопствен wrapper** (`envs/multi_agent_intersection.py`) околу
   `highway-env`-овата `intersection-v2`, кој ги декомпонира стандардно
   заедничките (cooperative) награди во **индивидуални награди по
   агент** - highway-env самиот не го поддржува ова низ стандардниот
   gymnasium интерфејс.
2. **VDN имплементација од нула** (`agents/vdn_agent.py`), директно
   базирана на теоријата од предметот (белешка 10 - "VDN и QMIX
   (decomposition на заедничката вредност за кооперативни тимови)").
3. **"Courtesy" (fairness) reward shaping** - додаден shaping член кој
   казнува агресивно возење блиску до human-driven возила
   (`courtesy_weight` во `envs/multi_agent_intersection.py`).
4. **Систематска, статистички ригорозна споредба** (повеќе seed-ови,
   95% доверителни интервали, паралелно стартување) на 3 пристапи +
   baseline, плус ablation, scalability и hyperparameter студии.
5. **Advanced дополнителна студија - 3 observation режими** (Kinematics /
   Pixels / Fusion, види секција подолу), тестирани и за IQL и за VDN,
   плус **robustness студија** (шум/blur/darken/branch-dropout наспроти
   clean евалуација). Главната тема на трудот си останува VDN/CTDE
   координацијата со Kinematics опсервации (точки 1-4 погоре) - ова е
   надградба, не замена.

## Структура на проектот

```
traffic-marl/
├── envs/
│   └── multi_agent_intersection.py   # wrapper (клучен придонес) - Kinematics/Pixels/Fusion obs_mode
├── agents/
│   ├── networks.py                   # Q-мрежа, Dueling/Pixel/Fusion Q-мрежи, replay buffers
│   ├── dqn_agent.py                  # DQN / Double DQN / Dueling DQN
│   ├── iql_manager.py                # Independent Q-Learning
│   ├── vdn_agent.py                  # VDN - CTDE пристап (клучен придонес)
│   └── heuristic_agent.py            # rule-based baseline
├── train.py                          # тренинг на еден метод (--obs_mode kinematics/pixels/fusion)
├── evaluate.py                       # евалуација на истрениран модел (+ опционален robustness perturb_fn)
├── experiments/
│   ├── parallel.py                   # паралелно стартување на seed-ови/конфигурации
│   ├── run_comparison.py             # главна споредба (сите методи × seed-ови, Kinematics)
│   ├── reward_ablation.py            # ablation: collision_reward, courtesy_weight
│   ├── scalability_study.py          # ablation: број контролирани возила
│   ├── hyperparameter_study.py       # ablation: learning_rate/gamma/epsilon
│   ├── significance_test.py          # Welch's t-test над comparison_summary.json
│   ├── observation_study.py          # advanced: Pixels/Fusion × IQL/VDN тренинг + споредба
│   ├── robustness.py                 # advanced: чисти perturbation функции (шум/blur/darken/dropout)
│   ├── robustness_study.py           # advanced: robustness евалуација на веќе-истренираните модели
│   ├── fusion_fix_study.py           # advanced: "modality collapse" наод + dropout/warm-start поправка
│   └── modality_sensitivity.py       # advanced: img_sens/kin_sens Q-вредностна дијагностика (Fusion)
├── report/
│   └── mdp_formalization.md          # формален MDP/Markov Game опис
├── tests/                            # unit + интеграциски тестови
├── visualize_results.py              # генерира 9 фигури за извештајот
└── results/                          # JSON резултати + фигури + зачувани модели
```

## Инсталација и брзо тестирање

```bash
pip install -r requirements.txt

python train.py --method vdn --episodes 200
python evaluate.py --method vdn --episodes 50
```

## Тестови

```bash
python -m unittest discover -s tests -v
```

66 unit + интеграциски тестови (`tests/test_core.py` - 34, оригиналниот
Kinematics/heuristic/IQL/VDN pipeline; `tests/test_observations.py` - 32,
Pixels/Fusion проширувањето), под 30 секунди вкупно - ги покриваат
replay buffer-ите, epsilon decay, courtesy shaping (вклучувајќи MC/DC-
изведени случаи и гранични вредности), `agg_stats` (вклучувајќи
fail-fast на празна листа), `tag`-именувањето на резултатски фајлови
(потребно за паралелно стартување, со гарантирано чистење дури и при
пад среде тест), VDN target-masking, double-DQN гранката (со
fault-revealing споредба, не само "не пука"), branch/boundary
покриеност на `HeuristicAgent`, дека наградите се РЕАЛНО различни по
агент (не само облик/тип), поправката на "истечена" награда по
пристигнување, дека ИСТ --seed дава ИДЕНТИЧНА тренинг-историја
(regression за баг: `random.sample()` во replay buffer-ите не беше
seed-иран - најдено при финалната ревизија, види `train.py::set_global_seed`),
`MultiAgentIntersectionEnv` преку вистински highway-env, **дека две
различни агентски pixel опсервации во ИСТ чекор НЕ се идентични**
(директно го наметнува "своја локална слика по агент" условот), Pixel/
Fusion Q-мрежите, `MultiAgentReplayBuffer` со fusion states, IQL/VDN
train_step() за pixels/fusion, robustness perturbation функциите
(шум/blur/darken/branch-dropout), modality-dropout/CNN-warm-start
поправката за Fusion, и дека `env.close()` СЕКОГАШ се повикува - дури и
кога `run_training()`/`evaluate()` ќе фрлат исклучок среде пат.

## Метрики за евалуација

- **mean_reward** - просечна награда по агент (± 95% доверителен интервал низ seed-ови)
- **collision_rate** - дел епизоди со барем еден судир (безбедност)
- **arrival_rate** - дел агенти кои успешно стигнале до дестинација (ефикасност)
- **avg_steps** - просечна должина на епизода (пропусна моќ)

## Observation режими (Kinematics / Pixels / Fusion) - advanced студија

Главната тема на трудот (истражувачкото прашање погоре) е VDN/CTDE
координацијата со **Kinematics** опсервации. Како advanced дополнителна
студија, истиот IQL/VDN pipeline (исти env поставки, seed-ови,
training/evaluation pipeline, метрики) е проширен со уште 2 типа
опсервација:

1. **Kinematics-only** (default, непроменето) - flatten вектор со
   [presence, x, y, vx, vy, cos_h, sin_h] за 6-те најблиски возила.
2. **Pixels-only** - секој агент добива **своја, локална, agent-centric
   сива (grayscale) слика** (стек од 4 фрејми, за CNN да може да "види"
   брзина преку разлика меѓу фрејмовите), обработена преку CNN. **Ова е
   намерна дизајн-одлука**: highway-env-овата `MultiAgentObservation`
   создава ОДДЕЛНА `GrayscaleObservation` инстанца по контролирано
   возило и ја центрира камерата токму на тоа возило
   (`observer_vehicle`) - секој агент реално гледа сопствена слика, НЕ
   ист глобален screenshot на цела раскрсница споделен меѓу сите (го
   потврдува `tests/test_observations.py::TestEnvPixelObservations`).
3. **Fusion** - локален kinematics вектор + локална сива слика, MLP
   гранка + CNN гранка споени пред финалните Q-вредности.

Архитектурите (`agents/networks.py`) и IQL/VDN алгоритамската логика
(double-DQN masking, VDN target-декомпозиција) се НАДГРАДЕНИ, не
заменети - истите постојни мрежи/replay buffer-и, само проширени со
`obs_mode`/`img_shape` параметри (не се користи Stable-Baselines3).

```bash
# тренинг/евалуација на еден метод со специфичен observation режим
python train.py --method vdn --episodes 200 --obs_mode pixels
python evaluate.py --method vdn --episodes 50 --obs_mode pixels

# целосна advanced студија: Pixels+Fusion × IQL/VDN (Kinematics×IQL/VDN
# се реупотребува од run_comparison.py, без retrain) + обединета табела
#
# --workers 8 (не default, кој би зел сите јадра-2): pixels/fusion replay
# buffer-ите чуваат слики (~100x поголеми од kinematics), премногу
# паралелни процеси лесно ја надминуваат RAM (тестирано: 8 workers
# безбедно на 31GB машина - прилагоди спрема сопствената RAM).
python experiments/observation_study.py --episodes 2000 --eval_episodes 200 --seeds 0 1 2 3 4 --workers 8
```

### Robustness студија

Истите веќе-истренирани модели (Kinematics/Pixels/Fusion × IQL/VDN) се
евалуираат под 3 типа нарушување на опсервацијата (СЕКОГАШ само во
евалуацијата - моделите се тренирани исклучиво на чисти опсервации):

- **kin_noise** - Gaussian шум на kinematics векторот (симулира
  непрецизни GPS/радар очитувања)
- **pixel_blur** / **pixel_dark** - Gaussian blur / затемнување на
  pixel сликата (магла/дожд, самрак)
- **drop_kin** / **drop_img** - привремено гасење на едната observation
  гранка (само Fusion - симулира испад на еден сензор)

Мерени метрики: mean_reward, collision_rate, arrival_rate, avg_steps, и
**% пад наспроти clean евалуација** (парирано по seed - истиот модел,
исти seed-ови, со и без нарушување).

```bash
python experiments/robustness_study.py --seeds 0 1 2 3 4 --eval_episodes 200
```

### "Modality collapse" наод и обид за поправка (`experiments/fusion_fix_study.py`)

Robustness студијата покажа необичен резултат: `pixel_blur`/`pixel_dark`
за Fusion излегоа со **~0% пад**, и покрај тоа што сликата реално се
менуваше. Директна инспекција на Q-вредностите (`experiments/modality_sensitivity.py`
- 30/30 метод×seed×агент комбинации, `agents/networks.py::drop_branch`)
покажа зошто: Fusion мрежата, тренирана на обичен начин, **речиси
целосно ја игнорира сликата** и се потпира само на kinematics
(Q-вредностите остануваат речиси непроменети дури и кога сликата е
целосно црна - `img_sens` ≈ 0.00-0.04, наспроти `kin_sens` ≈ 2.3-4.0
кога наместо тоа се нулира kinematics-гранката - т.е. mean ratio
kin/img ≈ 1100-2100×). Ова е познат феномен во multimodal learning
("modality collapse"/"gradient starvation") - кога едната гранка
(kinematics: чиста, ниско-димензионална, брзо-корисна) е многу полесна
за учење од другата (пиксели: на CNN-от му требаат многу повеќе
примери да извади нешто корисно од нив), мрежата брзо престанува да
учи преку потешката гранка.

`experiments/fusion_fix_study.py` тренира "поправена" Fusion варијанта
(`obs_mode="fusion"`, `modality_dropout_prob=0.3` + CNN
`pretrained_pixels_prefix` warm-start од Pixels-only модел, види
`train.py::run_training`) и ја споредува со оригиналната (сите бројки
подолу се од целосен 5-seed retrain, репродуцирачки преку командите
подолу):

| | img_sens (0=игнорира слика) | mean_reward | arrival_rate (clean) |
|---|---|---|---|
| Fusion (обичен) | ≈0.00-0.04 | 5.9-6.0 | 0.44-0.48 |
| Fusion (поправен) | 0.6-8.6 (просек по метод; поединечно 0-105, голема варијанса по агент/seed) | 4.8-5.5 | 0.21-0.36 |

Под `kin_noise`/`drop_kin`, обичниот Fusion губи arrival_rate 5-24%
(парирано по seed - директно во `results/robustness_study.json`). Кај
поправениот Fusion, arrival_rate под истите два услова **НЕ паѓа** -
всушност во просек РАСТЕ наспроти сопствената (пониска) clean основа
(IQL: +66% под `kin_noise`, +212% под `drop_kin`; VDN: +2% / +24%;
голема варијанса по seed, особено кај IQL).

Заклучок: поправката РЕАЛНО ја принудува мрежата да ја користи сликата
кај повеќето (но не сите) агент×seed комбинации - img_sens растe за
~2-3 реда на големина во просек, но со забележлива нестабилност (некои
агенти сепак остануваат близу colapse дури и со поправката). Тоа доаѓа
со цена - пониска чиста изведба (reward и arrival_rate паѓаат), во
замена за потенцијално поиздржливо однесување кога kinematics-сензорот
е нарушен/недостапен. Ова е класичен robustness-vs-performance
trade-off, не бесплатно подобрување - затоа "поправениот" Fusion НЕ го
заменува обичниот Fusion во главната споредба погоре, туку стои како
посебен, дополнителен наод.

```bash
# репродукција на img_sens/kin_sens табелата погоре
python experiments/modality_sensitivity.py --obs_modes fusion fusion_fixed --seeds 0 1 2 3 4
```

```bash
python experiments/fusion_fix_study.py --episodes 2000 --eval_episodes 200 --seeds 0 1 2 3 4 --workers 8
```

## Експерименти

Секој (метод, seed) / (конфигурација, seed) е независен job и се
пуштаат ПАРАЛЕЛНО преку `experiments/parallel.py` (повеќе CPU јадра
наместо еден по еден - `--workers N` за рачна контрола, `--workers 1`
за секвенцијално).

```bash
# главна споредба: heuristic / IQL / VDN, повеќе seed-ови
python experiments/run_comparison.py --episodes 1500 --eval_episodes 150 --seeds 0 1 2 3 4

# статистички тест (Welch's t-test) над comparison_summary.json
python experiments/significance_test.py

# ablation: тежина на collision_reward + courtesy_weight
python experiments/reward_ablation.py --episodes 800 --eval_episodes 80 --seeds 0 1 2

# scalability: 2-5 контролирани возила
python experiments/scalability_study.py --agent_counts 2 3 4 5 --episodes 800 --eval_episodes 80 --seeds 0 1 2

# хиперпараметарска студија (VDN): learning_rate / discount_factor / epsilon
python experiments/hyperparameter_study.py --episodes 800 --eval_episodes 80 --seeds 0 1 2
```

## Визуелизација

```bash
python visualize_results.py
```

Генерира 9 фигури во `results/figures/` (300 DPI, конзистентна палета
бои по метод): `training_curves.png`, `method_comparison.png`,
`radar_comparison.png`, `reward_ablation.png`, `courtesy_ablation.png`,
`scalability.png`, `hyperparameter_study.png`, `obs_mode_comparison.png`
(Kinematics/Pixels/Fusion × IQL/VDN), `robustness_study.png` (пад % под
секое нарушување). Секоја фигура за advanced студијата се прескокнува
без грешка ако соодветниот JSON сè уште не постои (`observation_study.py`/
`robustness_study.py` сè уште не се пуштени).
