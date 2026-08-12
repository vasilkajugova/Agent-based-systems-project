"""
Мал helper модул за паралелно стартување на независни тренинг+евалуација
"job"-ови (различни seed-ови, различни конфигурации) наместо секвенцијално
еден по еден.

Зошто ми е ова потребно: секој тренинг (train_and_eval за еден метод +
еден seed) е целосно независен од останатите - не делат состојба, само на
крајот ги комбинирам резултатите во една табела. Тоа значи можат да се
стартуваат ПАРАЛЕЛНО на различни CPU јадра, наместо да чекам едно по
друго. На машина со 16 јадра, 15 job-ови кои би траеле часови
секвенцијално, паралелно траат речиси исто колку еден единствен job.

Два детаља на кои морав да внимавам за ова реално да работи коректно:

1. **torch.set_num_threads(1) во секој worker процес.** PyTorch по
   default се обидува сам да користи повеќе CPU threads за сопствените
   операции. Ако пуштам 16 ПРОЦЕСИ (multiprocessing), а секој ВНАТРЕШНО
   исто така се обидува да зграби сите 16 јадра за torch операциите,
   резултатот е контра-продуктивен - процесите се "натпреваруваат" за
   истите јадра и вкупно испаѓа ПОБАВНО отколку секвенцијално. Затоа
   секој worker при стартување си поставува 1 thread - паралелизмот
   доаѓа од ПРОЦЕСИТЕ (различни job-ови), не од threads внатре во еден.

2. **Единствени имиња на резултатски фајлови по job** (види `tag`
   параметарот во train.py::run_training() и evaluate.py::evaluate()).
   Без ова, два паралелни процеси кои го тренираат ИСТИОТ метод (различни
   seed-ови) би пишувале во ИСТ results/{method}_history.json
   истовремено - race condition, а на Windows дури и PermissionError
   бидејќи фајлот е заклучен додека другиот процес пишува во него.

Употреба:
    from experiments.parallel import run_jobs
    results = run_jobs(my_job_function, list_of_job_args, max_workers=8)
    # my_job_function мора да е "top-level" функција (не lambda/closure) -
    # multiprocessing на Windows користи "spawn", кое бара job функцијата
    # и нејзините аргументи да можат да се "pickle"-аат (сериjaлизираат).
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np


def agg_stats(vals: list[float]) -> dict:
    """
    Средина + 95% доверителен интервал низ листа вредности (обично низ
    seed-ови). Ова ми се повторуваше буквално идентично во
    run_comparison.py, reward_ablation.py, scalability_study.py и
    hyperparameter_study.py, па го издвоив тука еднаш засекогаш - секоја
    скрипта само вика agg_stats(vals) наместо да ја копира истата
    формула.

    Враќа сурови броеви {"mean":..., "ci95":...}, НЕ веќе-форматиран
    стринг - форматирањето за принт секоја скрипта го прави одделно, само
    за приказ (види ги коментарите во run_comparison.py зошто е ова
    важно - visualize_results.py некогаш мораше да re-parse-ира
    веќе-форматирани стрингови, што е кревко).
    """
    if len(vals) == 0:
        # Одлука при ревизија на кодот: НАМЕРНО фрлам грешка тука наместо
        # тивко да вратам {"mean": nan, ...} (np.mean([]) би дало NaN + едно
        # RuntimeWarning, лесно за превидување). agg_stats() секогаш се вика
        # со листа изведена од --seeds (пр. [e["mean_reward"] for e in evals]) -
        # ако таа некогаш испадне празна, тоа секогаш значи погрешно
        # пресметана листа некаде погоре (пр. празен --seeds аргумент), не
        # валидна ситуација "нема резултати". Подобро е ГЛАСНО и ВЕДНАШ да
        # падне тука, отколку NaN тивко да се провлече низ JSON резултатите
        # и графиците, каде би бил многу потежок за најдување.
        raise ValueError("agg_stats() добив празна листа - нема вредности за агрегирање")
    m, s = float(np.mean(vals)), float(np.std(vals))
    ci = float(1.96 * s / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
    return {"mean": m, "ci95": ci}


def _init_worker():
    # се повикува еднаш, при create-ирање на секој worker процес
    import torch
    torch.set_num_threads(1)

    # Кирилски print() на не-UTF8 конзола фрла UnicodeEncodeError (истата
    # причина како во train.py, види коментар таму) - ВАЖНО е ова да се
    # постави тука, во самиот worker, не само во родителскиот процес:
    # secondmultiprocessing на Windows користи "spawn", секој worker
    # процес добива СОПСТВЕН sys.stdout со сопствен encoding, независен од
    # родителот (multiprocessing на Windows користи "spawn"). Затоа
    # reconfigure() во run_comparison.py/итн. не важи автоматски и за
    # worker-ите - мора одделно тука.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def default_worker_count() -> int:
    # оставам 1-2 јадра слободни за системот/останатите програми на
    # компјутерот, наместо да зграбам целосно сите достапни јадра
    cores = os.cpu_count() or 4
    return max(1, cores - 2)


def run_jobs(job_fn, jobs: list, max_workers: int | None = None, label: str = "job"):
    """
    Го стартува job_fn(job) за секој job од `jobs`, паралелно.

    Враќа листа резултати, ВО ИСТИОТ редослед како влезните `jobs`
    (не редослед на завршување) - за повикувачот полесно да ги спари
    резултатите со job-от од кој дошле.

    max_workers=1 ефективно се однесува секвенцијално (корисно за debug -
    полесно се чита распрснат print output кога сè оди еден по едно).
    """
    if max_workers is None:
        max_workers = default_worker_count()
    max_workers = min(max_workers, len(jobs)) if jobs else 1

    print(f"[parallel] стартувам {len(jobs)} {label}-и на {max_workers} паралелни процеси "
          f"({os.cpu_count()} CPU јадра достапни вкупно)")

    results = [None] * len(jobs)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker) as ex:
        future_to_idx = {ex.submit(job_fn, job): i for i, job in enumerate(jobs)}
        done = 0
        for fut in as_completed(future_to_idx):
            i = future_to_idx[fut]
            results[i] = fut.result()  # ако job_fn фрлил исклучок, тука повторно се крева - не сакам тивко да го проголтам
            done += 1
            print(f"[parallel] {done}/{len(jobs)} {label}-и завршени ({time.time() - t0:.0f}s поминато)")

    return results
