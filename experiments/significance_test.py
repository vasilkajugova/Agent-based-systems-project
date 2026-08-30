"""
Формален статистички тест (Welch's t-test) над веќе постоечките per-seed
резултати од experiments/run_comparison.py.

Зошто ми е ова потребно: во главната споредба (results/comparison_summary.json)
известувам средина ± 95% доверителен интервал низ seed-овите, и во извештајот/
тезата тврдам дека "IQL и VDN имаат статистички неразличлива просечна награда,
но значително различна стапка на судири/успех". Преклопување на 95% CI е добар
ВИЗУЕЛЕН индикатор, но не е формален статистички тест - затоа тука го
пресметувам Welch's t-test (не претпоставува еднакви варијанси помеѓу групите,
за разлика од обичен Student's t-test - соодветно тука бидејќи бројот на
епизоди/варијансата на VDN и IQL не мора да е ист) директно врз 5-те per-seed
вредности по метод, за секоја метрика посебно.

Мал број seed-ови (n=5 по метод) значи мала статистичка моќ - овој тест дава
дополнителна, построга потврда на наодот, не го заменува визуелниот преглед
на доверителните интервали во results/figures/method_comparison.png.

Употреба:
    python experiments/significance_test.py
    python experiments/significance_test.py --alpha 0.01
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scipy import stats

# Кирилски print() на не-UTF8 конзола фрла UnicodeEncodeError - истата
# причина како во train.py, види коментар таму.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

RESULTS_DIR = Path(__file__).parent.parent / "results"
METRICS = ["mean_reward", "collision_rate", "arrival_rate", "avg_steps"]
PAIRS = [("iql", "vdn"), ("iql", "heuristic"), ("vdn", "heuristic")]


def load_per_seed(summary_path: Path) -> dict[str, dict[str, list[float]]]:
    """Ги вчитува per-seed вредностите по метод и метрика од comparison_summary.json."""
    with open(summary_path, encoding="utf-8") as f:
        data = json.load(f)
    out: dict[str, dict[str, list[float]]] = {}
    for method, evals in data["per_seed"].items():
        out[method] = {m: [e[m] for e in evals] for m in METRICS}
    return out


def main(summary_path: Path, alpha: float):
    per_seed = load_per_seed(summary_path)

    print(f"\n===== Welch's t-test над per-seed резултатите (alpha={alpha}) =====")
    print(f"{'Споредба':<20}{'Метрика':<18}{'Средина A':<12}{'Средина B':<12}{'t':<10}{'df':<8}{'p':<10}{'Значајно?'}")

    results = []
    for a, b in PAIRS:
        if a not in per_seed or b not in per_seed:
            continue
        for metric in METRICS:
            vals_a, vals_b = per_seed[a][metric], per_seed[b][metric]
            res = stats.ttest_ind(vals_a, vals_b, equal_var=False)
            significant = bool(res.pvalue < alpha)
            row = {
                "a": a, "b": b, "metric": metric,
                "mean_a": sum(vals_a) / len(vals_a), "mean_b": sum(vals_b) / len(vals_b),
                "t": float(res.statistic), "df": float(res.df), "p": float(res.pvalue),
                "significant": significant,
            }
            results.append(row)
            print(
                f"{a + ' vs ' + b:<20}{metric:<18}{row['mean_a']:<12.4f}{row['mean_b']:<12.4f}"
                f"{row['t']:<10.3f}{row['df']:<8.2f}{row['p']:<10.4f}{'ДА' if significant else 'не'}"
            )

    out_path = RESULTS_DIR / "significance_test.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"alpha": alpha, "source": str(summary_path), "results": results}, f, indent=2, ensure_ascii=False)
    print(f"\nЗачувано во: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=str, default=str(RESULTS_DIR / "comparison_summary.json"))
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()
    main(Path(args.summary), args.alpha)
