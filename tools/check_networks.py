"""Diagnostyka: czy fenotyp (siec) uzywa wszystkich polaczen z genomu.

Przy feed_forward = False genom moze zawierac cykle i petle wlasne.
FeedForwardNetwork nie potrafi ulozyc takiego wezla w warstwy, wiec pomija go
po cichu - mutacja zmienia genom, a fenotyp jej nie widzi. Selekcja dziala
wtedy na czyms innym niz to, co dziedziczysz.

Skrypt liczy dla kazdej generacji, ile genomow ma siec:
  martwa   - zero polaczen, activate() zwraca same zera dla dowolnych wejsc
  niepelna - czesc polaczen z genomu nie trafila do sieci
  poprawna - fenotyp zgodny z genomem

Fitness jest losowy: badamy strukture genomow, nie zachowanie dronow.

Uzycie:
    python -m tools.check_networks conf/neat-cascade.txt
    python -m tools.check_networks conf/neat-e2e.txt --generations 30 --csv out.csv
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import random
from dataclasses import dataclass

import neat


@dataclass
class GenerationRow:
    generation: int
    dead: int
    partial: int
    correct: int
    total: int

    @property
    def dead_pct(self) -> float:
        return 100.0 * self.dead / self.total if self.total else 0.0

    @property
    def correct_pct(self) -> float:
        return 100.0 * self.correct / self.total if self.total else 0.0


def _count_links(net) -> int:
    """node_evals w neat-python 2.0.0: (node, act, agg, bias, response, links)."""
    return sum(len(node_eval[-1]) for node_eval in net.node_evals)


def _count_enabled(genome) -> int:
    return sum(1 for conn in genome.connections.values() if conn.enabled)


def survey(config_path: str, net_factory, generations: int, seed: int) -> list[GenerationRow]:
    """Uruchamia ewolucje i zbiera statystyki struktury sieci per generacja."""
    random.seed(seed)
    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path,
    )
    population = neat.Population(config)
    rows: list[GenerationRow] = []

    def evaluate(genomes, cfg) -> None:
        dead = partial = correct = 0
        for _, genome in genomes:
            net = net_factory(genome, cfg)
            in_genome = _count_enabled(genome)
            in_net = _count_links(net)
            if in_net == 0:
                dead += 1
            elif in_net != in_genome:
                partial += 1
            else:
                correct += 1
            genome.fitness = random.random()
        rows.append(GenerationRow(len(rows), dead, partial, correct, len(genomes)))

    # Reporter NEAT zasmieca stdout - wyciszamy go, wypisujemy wlasna tabele pozniej.
    with contextlib.redirect_stdout(io.StringIO()):
        population.run(evaluate, generations)

    return rows


def print_table(label: str, rows: list[GenerationRow], every: int = 5) -> None:
    print(f"\n=== {label} ===")
    print(f"{'gen':>5} {'martwe':>14} {'niepelne':>10} {'poprawne':>10}")
    for row in rows:
        if row.generation % every == 0 or row is rows[-1]:
            print(f"{row.generation:>5} {row.dead:>6} ({row.dead_pct:4.0f}%) "
                  f"{row.partial:>10} {row.correct:>10}")

    total = sum(r.total for r in rows)
    dead = sum(r.dead for r in rows)
    partial = sum(r.partial for r in rows)
    correct = sum(r.correct for r in rows)
    print(f"{'RAZEM':>5} {100 * dead / total:>11.1f}% "
          f"{100 * partial / total:>9.1f}% {100 * correct / total:>9.1f}%")


def write_csv(path: str, results: dict[str, list[GenerationRow]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["network_type", "generation", "dead", "partial", "correct", "total"])
        for label, rows in results.items():
            for row in rows:
                writer.writerow([label, row.generation, row.dead,
                                 row.partial, row.correct, row.total])
    print(f"\nZapisano: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sprawdza rozjazd genom - fenotyp w NEAT.")
    parser.add_argument("config", help="Sciezka do pliku conf/neat-*.txt")
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--every", type=int, default=5, help="Co ktora generacje wypisac wiersz")
    parser.add_argument("--csv", default=None, help="Opcjonalny plik wynikowy CSV")
    args = parser.parse_args()

    variants = {
        "FeedForwardNetwork": neat.nn.FeedForwardNetwork.create,
        "RecurrentNetwork": neat.nn.RecurrentNetwork.create,
    }

    results: dict[str, list[GenerationRow]] = {}
    for label, factory in variants.items():
        try:
            rows = survey(args.config, factory, args.generations, args.seed)
        except RuntimeError as exc:
            print(f"\n=== {label} ===\nPRZERWANE: {exc}")
            continue
        results[label] = rows
        print_table(label, rows, args.every)

    if args.csv and results:
        write_csv(args.csv, results)


if __name__ == "__main__":
    main()