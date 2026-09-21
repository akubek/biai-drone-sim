"""Porownuje przebiegi treningu zapisane w results/.

Liczy m.in. gladkosc krzywej fitnessu - srednia bezwzgledna zmiana miedzy
kolejnymi generacjami. Nizsza wartosc = mniej szumu ewaluacji (#15).

Uzycie:
    python tools/compare_runs.py results/run_a results/run_b
    python tools/compare_runs.py results/*            # w Git Bash
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RunSummary:
    name: str
    seed: str
    arch: str
    net_type: str
    scenarios: str
    generations: int
    evaluations: int
    jitter_best: float          # srednia |zmiana| best_fitness - ZALEZNA OD SKALI
    jitter_best_rel: float      # ta sama zmiana / sredni poziom - porownywalna miedzy przebiegami
    jitter_crash: float         # |zmiana| crash_rate - metryka ograniczona 0..1, wprost porownywalna
    jitter_dist: float
    mean_best: float
    jitter_mean: float
    final_best: float
    final_success: float
    final_crash: float
    final_dist: float
    total_time_s: float
    git_commit: str
    git_dirty: str


def _floats(rows: list[dict], column: str) -> list[float]:
    out = []
    for r in rows:
        value = r.get(column, "")
        if value not in ("", None):
            try:
                out.append(float(value))
            except ValueError:
                pass
    return out


def _jitter(values: list[float]) -> float:
    """Srednia bezwzgledna roznica miedzy kolejnymi wartosciami."""
    if len(values) < 2:
        return 0.0
    return statistics.fmean(abs(b - a) for a, b in itertools.pairwise(values))


def _jitter_rel(values: list[float]) -> float:
    """Jitter znormalizowany przez sredni poziom - bezwymiarowy.

    Bez tego przebieg o wyzszym fitnessie zawsze wyglada na mniej stabilny,
    bo wahania absolutne skaluja sie razem z wartosciami.
    """
    if len(values) < 2:
        return 0.0
    level = statistics.fmean(values)
    return _jitter(values) / level if abs(level) > 1e-9 else 0.0


def summarize(run_dir: Path) -> RunSummary | None:
    log = run_dir / "evolution_log.csv"
    if not log.exists():
        print(f"pomijam {run_dir.name}: brak evolution_log.csv", file=sys.stderr)
        return None

    with open(log, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"pomijam {run_dir.name}: pusty log", file=sys.stderr)
        return None

    manifest = {}
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    exp = manifest.get("exp_config", {})

    best = _floats(rows, "best_fitness")
    mean = _floats(rows, "mean_fitness")
    crash = _floats(rows, "crash_rate")
    dist = _floats(rows, "mean_min_dist_ratio")
    last = rows[-1]

    return RunSummary(
        name=run_dir.name,
        seed=str(exp.get("seed", last.get("seed", "?"))),
        arch=str(exp.get("arch", last.get("arch", "?"))),
        net_type=str(exp.get("net_type", last.get("net_type", "?"))),
        scenarios=str(exp.get("scenarios_per_genome", last.get("scenarios_per_genome", "?"))),
        generations=len(rows),
        evaluations=int(float(last.get("evaluations_total", 0) or 0)),
        jitter_best=_jitter(best),
        jitter_best_rel=_jitter_rel(best),
        jitter_crash=_jitter(crash),
        jitter_dist=_jitter(dist),
        mean_best=statistics.fmean(best) if best else 0.0,
        jitter_mean=_jitter(mean),
        final_best=best[-1] if best else 0.0,
        final_dist=dist[-1] if dist else 0.0,
        final_success=float(last.get("success_rate", 0) or 0),
        final_crash=float(last.get("crash_rate", 0) or 0),
        total_time_s=sum(_floats(rows, "wall_time_s")),
        git_commit=str(manifest.get("git_commit", "?")),
        git_dirty="TAK" if manifest.get("git_dirty") else "nie",
    )


def print_table(runs: list[RunSummary]) -> None:
    cols = [
        ("przebieg", lambda r: r.name[:34], 34),
        ("K", lambda r: r.scenarios, 3),
        ("arch", lambda r: r.arch, 8),
        ("siec", lambda r: r.net_type, 12),
        ("gen", lambda r: r.generations, 5),
        ("epizody", lambda r: r.evaluations, 9),
        ("skok wzgl", lambda r: f"{r.jitter_best_rel:.3f}", 10),
        ("skok crash", lambda r: f"{r.jitter_crash:.4f}", 11),
        ("skok dist", lambda r: f"{r.jitter_dist:.4f}", 10),
        ("sredni best", lambda r: f"{r.mean_best:.1f}", 12),
        ("best", lambda r: f"{r.final_best:.1f}", 9),
        ("sukces", lambda r: f"{r.final_success:.3f}", 7),
        ("kolizje", lambda r: f"{r.final_crash:.3f}", 8),
        ("dist konc", lambda r: f"{r.final_dist:.3f}", 10),
        ("czas[s]", lambda r: f"{r.total_time_s:.0f}", 8),
        ("commit", lambda r: r.git_commit, 9),
        ("dirty", lambda r: r.git_dirty, 6),
    ]

    header = "  ".join(f"{title:<{width}}" for title, _, width in cols)
    print(header)
    print("-" * len(header))
    for run in runs:
        print("  ".join(f"{fn(run)!s:<{width}}" for _, fn, width in cols))

    dirty = [r.name for r in runs if r.git_dirty == "TAK"]
    if dirty:
        print(f"\nUWAGA: przebiegi z niezacommitowanymi zmianami (nieodtwarzalne): {', '.join(dirty)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Porownuje przebiegi z results/.")
    parser.add_argument("runs", nargs="+", help="Foldery przebiegow")
    parser.add_argument("--sort", default="name",
                        choices=["name", "jitter", "best"],
                        help="Klucz sortowania tabeli")
    args = parser.parse_args()

    runs = [s for s in (summarize(Path(p)) for p in args.runs) if s is not None]
    if not runs:
        print("Brak przebiegow do porownania.", file=sys.stderr)
        sys.exit(1)

    key = {"name": lambda r: r.name,
           "jitter": lambda r: r.jitter_best_rel,
           "best": lambda r: -r.final_best}[args.sort]
    runs.sort(key=key)

    print_table(runs)


if __name__ == "__main__":
    main()