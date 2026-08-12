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

## Структура на проектот

```
traffic-marl/
├── envs/
│   └── multi_agent_intersection.py   # wrapper (клучен придонес)
├── agents/
│   ├── networks.py                   # Q-мрежа, Dueling Q-мрежа, replay buffers
│   ├── dqn_agent.py                  # DQN / Double DQN / Dueling DQN
│   ├── iql_manager.py                # Independent Q-Learning
│   ├── vdn_agent.py                  # VDN - CTDE пристап (клучен придонес)
│   └── heuristic_agent.py            # rule-based baseline
├── train.py                          # тренинг на еден метод
├── evaluate.py                       # евалуација на истрениран модел
├── experiments/
│   ├── parallel.py                   # паралелно стартување на seed-ови/конфигурации
│   ├── run_comparison.py             # главна споредба (сите методи × seed-ови)
│   ├── reward_ablation.py            # ablation: collision_reward, courtesy_weight
│   ├── scalability_study.py          # ablation: број контролирани возила
│   └── hyperparameter_study.py       # ablation: learning_rate/gamma/epsilon
├── report/
│   └── mdp_formalization.md          # формален MDP/Markov Game опис
├── tests/                            # unit + интеграциски тестови
├── visualize_results.py              # генерира 7 фигури за извештајот
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

31 unit + интеграциски тестови, под 10 секунди вкупно - ги покриваат
replay buffer-ите, epsilon decay, courtesy shaping (вклучувајќи MC/DC-
изведени случаи и гранични вредности), `agg_stats` (вклучувајќи
fail-fast на празна листа), `tag`-именувањето на резултатски фајлови
(потребно за паралелно стартување, со гарантирано чистење дури и при
пад среде тест), VDN target-masking, double-DQN гранката (со
fault-revealing споредба, не само "не пука"), branch/boundary
покриеност на `HeuristicAgent`, дека наградите се РЕАЛНО различни по
агент (не само облик/тип), поправката на "истечена" награда по
пристигнување, и `MultiAgentIntersectionEnv` преку вистински
highway-env.

## Метрики за евалуација

- **mean_reward** - просечна награда по агент (± 95% доверителен интервал низ seed-ови)
- **collision_rate** - дел епизоди со барем еден судир (безбедност)
- **arrival_rate** - дел агенти кои успешно стигнале до дестинација (ефикасност)
- **avg_steps** - просечна должина на епизода (пропусна моќ)

## Експерименти

Секој (метод, seed) / (конфигурација, seed) е независен job и се
пуштаат ПАРАЛЕЛНО преку `experiments/parallel.py` (повеќе CPU јадра
наместо еден по еден - `--workers N` за рачна контрола, `--workers 1`
за секвенцијално).

```bash
# главна споредба: heuristic / IQL / VDN, повеќе seed-ови
python experiments/run_comparison.py --episodes 1500 --eval_episodes 150 --seeds 0 1 2 3 4

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

Генерира 7 фигури во `results/figures/` (300 DPI, конзистентна палета
бои по метод): `training_curves.png`, `method_comparison.png`,
`radar_comparison.png`, `reward_ablation.png`, `courtesy_ablation.png`,
`scalability.png`, `hyperparameter_study.png`.
