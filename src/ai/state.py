from src.core.environment import TIERS, Scenario, generate_scenario_for_tier, generate_scenarios
from src.core.stats import EpisodeResult


class TrainingState:
    """Central source of truth about the training state and mode."""
    
    def __init__(self, exp_config: dict | None = None):
        if exp_config is None:
            exp_config = {}
        self.exp_config = exp_config
        # 0=Pure NEAT, 1=Linear Expert, 2=Blind Curriculum, 3=Full Curriculum
        self.mode = exp_config.get("training_mode", 0)

        # Retrieving hyperparameters
        self.max_help_gens = exp_config.get("max_help_gens", 100)
        self.start_weight = exp_config.get("start_weight", 0.85)
        self.scenarios_per_genome = exp_config.get("scenarios_per_genome", 6)
        self.rotate_every = exp_config.get("scenario_rotate_every", 3)

        # Variables tracking progress
        self.generation = 0
        self.current_tier = exp_config.get("start_tier", 1)
        self.curriculum_enabled = exp_config.get("curriculum", True)
        if self.current_tier not in TIERS:
            raise ValueError(f"start_tier={self.current_tier} does not exist in TIERS {sorted(TIERS)}")
        self.last_success_rate = 0.0

        self.current_help_weight = 0.0

        self.last_metrics: dict[int, EpisodeResult] = {}
        # Immediately initialize parameters for the 0th generation
        self._scenarios: list[Scenario] | None = None
        self._scenarios_tier: int | None = None
        self._gens_since_rotate: int = -1
        self.update_parameters()

    @property
    def num_obstacles(self) -> int:
        return TIERS[self.current_tier]["obstacles"]

    def update_parameters(self):
        """Calculates parameters based on the selected mode (0-3), stage, and generation."""

        if self.mode in [0, 2]:
            # Tryby bez eksperta
            self.current_help_weight = 0.0

        elif self.mode == 1:
            # Tryb 1: Liniowy spadek
            # Zastosowano twardą podłogę 0.15, żeby drony nie zginęły z dnia na dzień
            drop_per_gen = self.start_weight / self.max_help_gens
            self.current_help_weight = max(0.15, self.start_weight - (self.generation * drop_per_gen))

            #TODO na razie po max help gen tez wylaczamy - do zmiany na rozpoznanie czy drony maja dobry success rate
            if self.generation > self.max_help_gens:
                self.current_help_weight = 0.0
            #bezwzględnie wyłączamy pomoc po 150 generacjach
            if self.generation > 150:
                self.current_help_weight = 0.0

        elif self.mode == 3:
            # Przemapowane z 3 etapow na 6 poziomow.
            if self.current_tier <= 2:
                self.current_help_weight = max(
                    0.25, self.start_weight - (self.generation / 40.0)
                )
            elif self.current_tier <= 4:
                self.current_help_weight = 0.15
            else:
                self.current_help_weight = 0.0

    def scenarios_for_generation(self) -> list[Scenario]:
        if self._scenarios is None or self._scenarios_tier != self.current_tier:
            self._scenarios = generate_scenarios(self.scenarios_per_genome,
                                                self.current_tier)
            self._scenarios_tier = self.current_tier
            self._gens_since_rotate = 0
        elif self._gens_since_rotate >= self.rotate_every:
            self._scenarios = (self._scenarios[1:]
                            + [generate_scenario_for_tier(self.current_tier)])
            self._gens_since_rotate = 0
        self._gens_since_rotate += 1
        return self._scenarios