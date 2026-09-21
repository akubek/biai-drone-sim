"""Draws the holdout set as contact sheets in PNG format - one per difficulty level.

Serves to visually check whether the levels actually differ in difficulty,
before freezing data/holdout.json.

Usage (from the project directory):
    python -m tools.draw_holdout
    python -m tools.draw_holdout --out docs/holdout --cols 5
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")   # rysujemy bez okna

import pygame

BG = (18, 22, 28)
PANEL = (28, 34, 42)
OBSTACLE = (150, 50, 50)
OBSTACLE_EDGE = (255, 100, 100)
TARGET = (0, 220, 90)
START = (0, 190, 255)
PATH = (90, 100, 120)
LABEL = (210, 215, 225)


def draw_thumbnail(scenario: dict, size: tuple[int, int],
                   world: tuple[int, int], target_radius_px: int) -> pygame.Surface:
    """Jeden scenariusz przeskalowany do miniatury."""
    surf = pygame.Surface(size)
    surf.fill(PANEL)

    sx = size[0] / world[0]
    sy = size[1] / world[1]

    def to_px(p):
        return (int(p[0] * sx), int(p[1] * sy))

    for x, y, w, h in scenario["obstacles_px"]:
        rect = pygame.Rect(int(x * sx), int(y * sy), max(1, int(w * sx)), max(1, int(h * sy)))
        pygame.draw.rect(surf, OBSTACLE, rect)
        pygame.draw.rect(surf, OBSTACLE_EDGE, rect, 1)

    start = to_px(scenario["start_px"])
    target = to_px(scenario["target_px"])

    pygame.draw.line(surf, PATH, start, target, 1)
    pygame.draw.circle(surf, TARGET, target, max(3, int(target_radius_px * sx)), 1)
    pygame.draw.circle(surf, TARGET, target, 2)
    pygame.draw.circle(surf, START, start, 4)

    pygame.draw.rect(surf, (60, 68, 80), surf.get_rect(), 1)
    return surf


def build_sheet(scenarios: list[dict], tier: int, cols: int,
                world: tuple[int, int], target_radius_px: int,
                thumb_w: int = 300) -> pygame.Surface:
    thumb_h = int(thumb_w * world[1] / world[0])
    pad = 8
    label_h = 34
    rows = math.ceil(len(scenarios) / cols)

    width = cols * thumb_w + (cols + 1) * pad
    height = label_h + rows * (thumb_h + pad) + pad
    sheet = pygame.Surface((width, height))
    sheet.fill(BG)

    font = pygame.font.SysFont("arial", 18)
    dists = [math.hypot(s["target_px"][0] - s["start_px"][0],
                        s["target_px"][1] - s["start_px"][1]) / 200.0
             for s in scenarios]
    obs = [len(s["obstacles_px"]) for s in scenarios]
    caption = (f"Poziom {tier}   |   n={len(scenarios)}   |   "
               f"dystans {min(dists):.2f}-{max(dists):.2f} m (sr. {statistics.fmean(dists):.2f})   |   "
               f"przeszkody {min(obs)}-{max(obs)}")
    sheet.blit(font.render(caption, True, LABEL), (pad, pad))

    for i, scenario in enumerate(scenarios):
        col, row = i % cols, i // cols
        x = pad + col * (thumb_w + pad)
        y = label_h + pad + row * (thumb_h + pad)
        sheet.blit(draw_thumbnail(scenario, (thumb_w, thumb_h), world, target_radius_px), (x, y))

    return sheet


def main() -> None:
    parser = argparse.ArgumentParser(description="Rysuje holdout do PNG.")
    parser.add_argument("--holdout", default="data/holdout.json")
    parser.add_argument("--out", default="docs/holdout")
    parser.add_argument("--cols", type=int, default=5)
    parser.add_argument("--width", type=int, default=1024, help="Szerokosc swiata w px")
    parser.add_argument("--height", type=int, default=720, help="Wysokosc swiata w px")
    parser.add_argument("--target-radius", type=int, default=50)
    args = parser.parse_args()

    # Bez pygame.init() - nie potrzebujemy okna ani audio, tylko Surface i czcionki.
    pygame.font.init()

    blob = json.loads(Path(args.holdout).read_text(encoding="utf-8"))
    by_tier: dict[int, list[dict]] = defaultdict(list)
    for scenario in blob["scenarios"]:
        by_tier[int(scenario["tier"])].append(scenario)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    world = (args.width, args.height)

    print(f"{'poziom':>7} {'n':>4} {'dystans min-max [m]':>22} {'sredni':>8} {'przeszkody':>12}")
    for tier in sorted(by_tier):
        scenarios = by_tier[tier]
        sheet = build_sheet(scenarios, tier, args.cols, world, args.target_radius)
        path = out_dir / f"holdout_tier{tier}.png"
        pygame.image.save(sheet, str(path))

        dists = [math.hypot(s["target_px"][0] - s["start_px"][0],
                            s["target_px"][1] - s["start_px"][1]) / 200.0
                 for s in scenarios]
        obs = [len(s["obstacles_px"]) for s in scenarios]
        print(f"{tier:>7} {len(scenarios):>4} {min(dists):>10.2f} - {max(dists):<9.2f} "
              f"{statistics.fmean(dists):>8.2f} {min(obs):>6} - {max(obs):<5}")

    print(f"\nzapisano {len(by_tier)} arkuszy do {out_dir}/")


if __name__ == "__main__":
    main()