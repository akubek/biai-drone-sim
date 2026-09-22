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
    jitter_dist: float          # srednia |zmiana| mean_min_dist_ratio miedzy generacjami
    jitter_best_rel: float      # jitter best_fitness / sredni poziom - bezwymiarowy
    dist: float                 # ponizsze: srednie z ostatnich N generacji
    crash: float
    timeout: float
    escape: float
    stagnation: float
    success: float
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
    if len(values) < 2:
        return 0.0
    return statistics.fmean(abs(b - a) for a, b in zip(values, values[1:]))
 
 
def _jitter_rel(values: list[float]) -> float:
    """Jitter znormalizowany przez sredni poziom - porownywalny miedzy skalami."""
    if len(values) < 2:
        return 0.0
    level = statistics.fmean(values)
    return _jitter(values) / level if abs(level) > 1e-9 else 0.0
 
 
def _tail_mean(rows: list[dict], column: str, tail: int) -> float:
    values = _floats(rows[-tail:], column)
    return statistics.fmean(values) if values else 0.0
 
 
def summarize(run_dir: Path, tail: int) -> RunSummary | None:
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
    last = rows[-1]
 
    return RunSummary(
        name=run_dir.name,
        seed=str(exp.get("seed", last.get("seed", "?"))),
        arch=str(exp.get("arch", last.get("arch", "?"))),
        net_type=str(exp.get("net_type", last.get("net_type", "?"))),
        scenarios=str(exp.get("scenarios_per_genome", last.get("scenarios_per_genome", "?"))),
        generations=len(rows),
        evaluations=int(float(last.get("evaluations_total", 0) or 0)),
        jitter_dist=_jitter(_floats(rows, "mean_min_dist_ratio")),
        jitter_best_rel=_jitter_rel(_floats(rows, "best_fitness")),
        dist=_tail_mean(rows, "mean_min_dist_ratio", tail),
        crash=_tail_mean(rows, "crash_rate", tail),
        timeout=_tail_mean(rows, "timeout_rate", tail),
        escape=_tail_mean(rows, "escape_rate", tail),
        stagnation=_tail_mean(rows, "stagnation_rate", tail),
        success=_tail_mean(rows, "success_rate", tail),
        total_time_s=sum(_floats(rows, "wall_time_s")),
        git_commit=str(manifest.get("git_commit", "?")),
        git_dirty="TAK" if manifest.get("git_dirty") else "nie",
    )
 
 
def print_table(runs: list[RunSummary], tail: int) -> None:
    cols = [
        ("przebieg", lambda r: r.name[-24:], 24),
        ("K", lambda r: r.scenarios, 2),
        ("arch", lambda r: r.arch[:7], 7),
        ("siec", lambda r: r.net_type[:4], 4),
        ("gen", lambda r: r.generations, 4),
        ("dist", lambda r: f"{r.dist:.3f}", 6),
        ("crash", lambda r: f"{r.crash:.3f}", 6),
        ("timeout", lambda r: f"{r.timeout:.3f}", 7),
        ("escape", lambda r: f"{r.escape:.3f}", 6),
        ("stagn", lambda r: f"{r.stagnation:.3f}", 6),
        ("sukces", lambda r: f"{r.success:.3f}", 6),
        ("skok dist", lambda r: f"{r.jitter_dist:.4f}", 9),
        ("skok fit", lambda r: f"{r.jitter_best_rel:.3f}", 8),
        ("czas", lambda r: f"{r.total_time_s:.0f}", 5),
        ("commit", lambda r: r.git_commit[:7], 7),
        ("dirty", lambda r: r.git_dirty, 5),
    ]
 
    print(f"\nWartosci 'dist'..'sukces' to srednie z ostatnich {tail} generacji.\n")
    header = " ".join(f"{t:<{w}}" for t, _, w in cols)
    print(header)
    print("-" * len(header))
    for run in runs:
        print(" ".join(f"{str(fn(run)):<{w}}" for _, fn, w in cols))
 
    dirty = [r.name for r in runs if r.git_dirty == "TAK"]
    if dirty:
        print(f"\nUWAGA: nieodtwarzalne (niezacommitowane zmiany): {len(dirty)} przebiegow")
 
 
def main() -> None:
    parser = argparse.ArgumentParser(description="Porownuje przebiegi z results/.")
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--tail", type=int, default=10,
                        help="Z ilu ostatnich generacji liczyc srednie koncowe.")
    parser.add_argument("--sort", default="name", choices=["name", "dist", "crash"])
    args = parser.parse_args()
 
    runs = [s for s in (summarize(Path(p), args.tail) for p in args.runs) if s is not None]
    if not runs:
        print("Brak przebiegow do porownania.", file=sys.stderr)
        sys.exit(1)
 
    key = {"name": lambda r: r.name,
           "dist": lambda r: r.dist,
           "crash": lambda r: r.crash}[args.sort]
    runs.sort(key=key)
    print_table(runs, args.tail)
 
 
if __name__ == "__main__":
    main()
