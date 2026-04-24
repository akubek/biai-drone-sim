import neat
import pygame
import math
import sys
import random

from .drone import Drone
from .constants import *


def remove_drone(
    index: int,
    drones: list[Drone],
    nets: list[neat.nn.FeedForwardNetwork],
    ge: list[neat.DefaultGenome],
) -> None:
    """Bezpiecznie usuwa drona, jego sieć i genom z list aktywnych obiektów."""
    drones.pop(index)
    nets.pop(index)
    ge.pop(index)


def eval_genomes(genomes, config) -> None:
    """
    Funkcja oceniająca (Fitness Function). Wywoływana raz na generację.
    Dostaje listę genomów (mózgów) i musi każdemu przypisać ocenę (fitness).
    """
    # Pobieramy aktywne okno Pygame (żeby nie inicjować go co generację)
    screen = pygame.display.get_surface()
    clock = pygame.time.Clock()

    nets = []
    drones = []
    ge = []

    # 1. TWORZENIE POPULACJI
    for genome_id, genome in genomes:
        # Resetujemy ocenę na start
        genome.fitness = 0.0

        # Tworzymy sieć neuronową (FeedForward) na podstawie genomu
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)

        new_drone = Drone(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
        new_drone.hover_frames = 0
        # Tworzymy fizycznego drona w locie
        drones.append(new_drone)
        ge.append(genome)

    # 2. LOSOWANIE CELU DLA GENERACJI
    # Każde pokolenie ma inny cel, żeby drony uczyły się "szukać", a nie "pamiętać trasę"
    target_pos = (
        random.randint(100, SCREEN_WIDTH - 100),
        random.randint(100, SCREEN_HEIGHT - 100),
    )

    run = True
    frames = 0
    max_frames = FPS * 15  # Limit czasu na próbę: 15 sekund przy 60 FPS

    # 3. GŁÓWNA PĘTLA SYMULACJI DLA DANEJ GENERACJI
    while run and len(drones) > 0:
        frames += 1

        # Koniec czasu dla tej generacji
        if frames > max_frames:
            break

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        screen.fill((20, 25, 30))

        # Rysujemy cel na ekranie
        pygame.draw.circle(screen, (0, 255, 0), target_pos, 15, 2)
        pygame.draw.circle(screen, (0, 255, 0), target_pos, 3)

        # Używamy tej listy, by zapisać indeksy dronów do usunięcia
        # (Nie możemy usuwać elementów z listy po której właśnie iterujemy)
        drones_to_remove = []

        for i, drone in enumerate(drones):
            # A. Pobranie danych ze środowiska (Oczy)
            inputs = drone.get_inputs(
                target_pos[0], target_pos[1], SCREEN_WIDTH, SCREEN_HEIGHT
            )

            # B. Decyzja sieci neuronowej (Mózg)
            # Funkcja aktywacji to Sigmoid, więc zwróci wartości od 0.0 do 1.0
            output = nets[i].activate(inputs)
            l_thrust = output[0]
            r_thrust = output[1]

            # C. Wykonanie ruchu (Mięśnie)
            drone.apply_thrust(l_thrust, r_thrust)
            drone.update()

            # D. LOGIKA FITNESS (Ocena)
            dist = math.hypot(drone.x - target_pos[0], drone.y - target_pos[1])

            # 1. NAGRODA ZA "MĄDRE" PRZEŻYCIE (Zależna od odległości)
            # Ustawiamy strefę np. 400 pikseli od celu. Jeśli dron jest poza nią, nie dostaje nic.
            # Im głębiej wlatuje w tę strefę, tym szybciej nabija mu się licznik punktów za czas.
            scale_S = 50.0
            proximity_multiplier = 1.0 / (1.0 + (dist / scale_S) ** 2)

            time_progress = frames / max_frames
            time_decay = 1.0 - (0.8 * time_progress)

            if dist < 250:
                ge[i].fitness += proximity_multiplier * time_decay * 2.0

            if dist > 300:
                ge[i].fitness -= 0.1 - 0.05 * time_progress

            # WARUNKI KOŃCOWE DRONA
            # 2. Śmierć (Zderzenie ze ścianą)
            if (
                drone.x < 0
                or drone.x > SCREEN_WIDTH
                or drone.y < 0
                or drone.y > SCREEN_HEIGHT
            ):
                ge[i].fitness -= 10  # Bolesna kara za zderzenie
                drones_to_remove.append(i)

            # 3. Sprawdzanie strefy zawiśnięcia (Złoty środek)
            else:
                if dist < 40:
                    drone.hover_frames += 1

                    base_hover_reward = 2.0
                    growth_factor = 0.1

                    current_reward = (
                        base_hover_reward
                        * time_decay
                        * (1 + drone.hover_frames * growth_factor)
                    )
                    ge[i].fitness += current_reward
                    # Gigantyczna ciągła nagroda za samo przebywanie w kółku

                    # Sukces - wyhamował i ustał 1 sekundę (60 klatek)
                    if drone.hover_frames >= FPS * 1:
                        ge[i].fitness += 2000 * time_decay
                        # Mnożnik za czas - im szybciej to zrobił, tym lepiej
                        ge[i].fitness += (max_frames - frames) * 2
                        drones_to_remove.append(i)
                        continue  # Przeskakujemy rysowanie, bo dron znika w blasku chwały

                # Jeśli z kółka wyleciał - licznik hamowania wraca na zero
                else:
                    drone.hover_frames = 0

                # Rysowanie żywego drona
                drone.draw(screen, target_pos, l_thrust, r_thrust)
        # Odwracamy listę indeksów i usuwamy martwe drony
        # (Odwrócenie chroni nas przed przesunięciem indeksów przy używaniu pop())
        for i in reversed(drones_to_remove):
            remove_drone(i, drones, nets, ge)

        pygame.display.flip()
        clock.tick(0)


def run_neat(config_path: str) -> None:
    """Konfiguruje i uruchamia algorytm NEAT."""
    # Inicjujemy Pygame przed startem ewolucji
    pygame.init()
    pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("BIAI Drone Sim - AI Evolution")

    # Wczytanie konfiguracji z pliku
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    # Tworzenie populacji (np. 50 dronów)
    population = neat.Population(config)

    # Dodanie reporterów (wypisują statystyki do konsoli)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    # START EWOLUCJI
    # Uruchamiamy na maksymalnie 100 generacji
    print("🧠 Starting neuroevolution...")
    winner = population.run(eval_genomes, 100)

    # Po zakończeniu treningu
    print(f"\n🏆 Best genome found:\n{winner}")
    pygame.quit()
