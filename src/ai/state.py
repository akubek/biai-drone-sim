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
        self.target_obstacles = exp_config.get("target_obstacles", 5)
        self.start_weight = exp_config.get("start_weight", 0.85)
        self.scenarios_per_genome = exp_config.get("scenarios_per_genome", 3)

        # Variables tracking progress
        self.generation = 0
        self.current_stage = 1
        self.last_success_rate = 0.0

        # Working values (will be overwritten shortly)
        self.current_help_weight = 0.0
        self.num_obstacles = 0

        self.last_metrics: dict[int, EpisodeResult] = {}
        # Immediately initialize parameters for the 0th generation
        self.update_parameters()

    def update_parameters(self):
        """Calculates parameters based on the selected mode (0-3), stage, and generation."""

        # ==========================================
        # 1. MAP DIFFICULTY (Obstacles)
        # ==========================================
        if self.mode in [2, 3]:  # Tryby korzystające z etapów (Curriculum)
            if self.current_stage == 1:
                self.num_obstacles = 0
            elif self.current_stage == 2:
                # Połowa docelowych przeszkód
                self.num_obstacles = max(1, self.target_obstacles // 2)
            else:  # Etap 3
                self.num_obstacles = self.target_obstacles
        else:
            # Tryby sztywne (0 i 1) - od razu docelowa mapa
            self.num_obstacles = self.target_obstacles

        # ==========================================
        # 2. EXPERT HELP (Action Blending)
        # ==========================================
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
            # Tryb 3: Pełne Curriculum (Wygaszanie zgrane z etapami)
            if self.current_stage == 1:
                # Szybki spadek na pustej mapie, ale zatrzymuje się na 0.25 (bezpieczna asysta)
                self.current_help_weight = max(0.25, self.start_weight - (self.generation / 40.0))
            elif self.current_stage == 2:
                # Dodaliśmy przeszkody, więc dajemy lekką pomoc
                self.current_help_weight = 0.15 
            else:
                # Etap 3 - Absolutna samodzielność
                self.current_help_weight = 0.0