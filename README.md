# Autonomous 2D Drone Navigation Using Neuroevolution

**Status: Active Development (Work in Progress)**

This repository contains an academic project developed for the Biologically Inspired Artificial Intelligence (BIAI) course at the Silesian University of Technology. The project explores teaching an autonomous 2D drone to navigate through a simulated obstacle course using the NEAT (NeuroEvolution of Augmenting Topologies) algorithm.

## Authors
* **Artur Kubek** (Lead Developer)
* **Maciej Guja** (Co-author)

## Project Overview

The primary goal of this project is to demonstrate how complex spatial navigation capabilities can emerge in a neural network through neuroevolution. The drone evolves, generation by generation, to navigate randomized environments, avoid obstacles, reach a target destination, and successfully hover above it for 1.5 seconds.

The project features a custom 2D simulation environment with realistic SI-unit physics. The drone perceives its surroundings using 8 simulated distance rays, alongside proprioceptive data (velocity, angle, and distance to the target). 

During the evolutionary process, a population of 512 drones is simulated in each generation, with the fittest individuals selected for reproduction and mutation over 200+ generations.

## Neural Network Architectures

We are actively testing and benchmarking two different control architectures:

1. **Cascade Architecture (Current Best):** 
   * **Flow:** NEAT Network -> Target Angle/Thrust -> PID Flight Controller -> Motors.
   * **Result:** The network delegates low-level motor stabilization to a classic PID controller. This approach has proven to be highly stable, easier to train, and yields significantly higher fitness scores.
2. **End-to-End (E2E) Architecture:** 
   * **Flow:** NEAT Network -> Left/Right Motor Thrust directly.
   * **Result:** The network must learn both high-level pathfinding and low-level stabilization simultaneously. This approach is much harder to train, resulting in lower success rates and less stable flight behavior.

## Development Phases & Roadmap

The project is structured into three main phases. We have completed the baseline testing and are currently implementing advanced optimization techniques.

### Phase 1: Baseline (Completed)
* Implementation of the custom 2D physics engine and drone dynamics.
* Basic NEAT integration on an empty map.
* Initial benchmarking of Cascade vs. End-to-End architectures.

### Phase 2: Obstacles & Expert Guidance (In Progress)
* Introduction of a randomized obstacle generator.
* Implementation of an algorithmic "Expert" drone model with programmed behavior.
* Testing expert help via signal mixing.
* *Key Finding:* Input mixing (blending expert signals with agent signals) proved ineffective. It did not provide a meaningful learning signal, and agents failed to reliably learn obstacle avoidance in complex maps.

### Phase 3: Optimization & Curriculum Learning (Planned)
* **Expert-Guided Fitness Shaping:** Replacing input mixing with reward-based shaping. Agents will be rewarded for behavior that aligns with the expert's path rather than directly blending control signals.
* **Curriculum Learning:** Structuring the training process to start on an empty map and progressively increase the obstacle count and environmental complexity as the population's fitness improves.
* **Hyperparameter Tuning:** Exploring a broader range of NEAT configurations (population size, mutation rates, speciation thresholds) to overcome current performance plateaus.

## Technology Stack

* **Language:** Python
* **Neuroevolution:** NEAT-Python
* **Simulation & Visualization:** Custom 2D renderer using Pygame / Math libraries

## Running the Simulation

*Note: Because this project is in active development, execution commands and configuration structures are subject to change.*

1. Clone the repository.
2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the main launcher script:

  ```bash
  python launcher.py
  ```

4. Configuration files for the NEAT algorithm (`neat-cascade.txt` and `neat-e2e.txt`) can be found and modified in the `conf/` directory. Physics and reward parameters are defined within the `src/config/` module.
