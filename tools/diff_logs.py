"""Compares two evolution_log.csv files based on COMMON columns.

Usage:
    python tools/diff_logs.py results/before/evolution_log.csv results/after/evolution_log.csv
"""
import csv
import sys

# Kolumny zmienne z natury - pomijamy.
IGNORE = {"run_id", "wall_time_s"}


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    a, b = load(sys.argv[1]), load(sys.argv[2])
    if not a or not b:
        print("pusty plik"); return

    common = [c for c in a[0] if c in b[0] and c not in IGNORE]
    only_a = [c for c in a[0] if c not in b[0]]
    only_b = [c for c in b[0] if c not in a[0]]
    print(f"rows: {len(a)} vs {len(b)} | common columns: {len(common)}")
    if only_a: print(f"only in the first: {only_a}")
    if only_b: print(f"only in the second:   {only_b}")

    diffs = 0
    for i, (ra, rb) in enumerate(zip(a, b)):
        for c in common:
            if ra[c] != rb[c]:
                print(f"  gen {i}, {c}: {ra[c]!r} != {rb[c]!r}")
                diffs += 1
                if diffs > 30:
                    print("  ... (more differences skipped)"); return
    print("IDENTICAL" if diffs == 0 and len(a) == len(b) else f"{diffs} differences")


if __name__ == "__main__":
    main()