import json

from src.ai.curriculum import CurriculumController


class _Holdout:
    def __init__(self):
        self.last_by_tier = {}


class _State:
    def __init__(self, tier=1):
        self.current_tier = tier
        self.curriculum_enabled = True
        self.resets = 0

    def force_scenario_reset(self):
        self.resets += 1


class _Sp:
    def __init__(self):
        self.last_improved = 0
        self.fitness_history = [1.0, 1.0]


class _Species:
    def __init__(self, n=3):
        self.species = {i: _Sp() for i in range(n)}


RULES = {
    "promote": {"expert_fraction": 0.70, "crash_floor": 0.20, "consecutive": 2},
    "demote": {"expert_fraction": 0.25, "consecutive": 2},
    "min_dwell_gens": 20,
    "cooldown_after_demote_gens": 30,
}
# odpowiada zmierzonym wynikom eksperta
BASELINES = {"expert": {
    "1": {"success": 1.00, "crash": 0.00}, "2": {"success": 1.00, "crash": 0.00},
    "3": {"success": 0.70, "crash": 0.30}, "4": {"success": 0.40, "crash": 0.60},
    "5": {"success": 0.25, "crash": 0.75}, "6": {"success": 0.15, "crash": 0.85},
}}


def _make(tmp_path, state, holdout, name="r"):
    r, b = tmp_path / f"{name}_rules.json", tmp_path / f"{name}_base.json"
    r.write_text(json.dumps(RULES))
    b.write_text(json.dumps(BASELINES))
    return CurriculumController(state, holdout, str(r), str(b))


def _tick(ctrl, gen, species, success, crash=0.0):
    ctrl.start_generation(gen)
    ctrl.holdout.last_by_tier = {
        ctrl.state.current_tier: {"success_rate": success, "crash_rate": crash}
    }
    ctrl.post_evaluate(None, {}, species, object())

def test_promotion_requires_two_consecutive(tmp_path):
    st, sp = _State(1), _Species()
    c = _make(tmp_path, st, _Holdout())
    _tick(c, 20, sp, 0.80)
    assert st.current_tier == 1          # jeden dobry odczyt to za malo
    _tick(c, 30, sp, 0.80)
    assert st.current_tier == 2


def test_no_promotion_before_min_dwell(tmp_path):
    st, sp = _State(1), _Species()
    c = _make(tmp_path, st, _Holdout())
    _tick(c, 0, sp, 1.0)
    _tick(c, 10, sp, 1.0)
    assert st.current_tier == 1          # min_dwell=20 jeszcze nie minal
    _tick(c, 20, sp, 1.0)
    assert st.current_tier == 2


def test_threshold_scales_with_expert(tmp_path):
    # tier 4: ekspert 0.40 -> prog 0.28, wiec 0.30 wystarcza
    st = _State(4)
    c = _make(tmp_path, st, _Holdout(), "a")
    _tick(c, 20, _Species(), 0.30, crash=0.50)
    _tick(c, 30, _Species(), 0.30, crash=0.50)
    assert st.current_tier == 5

    # tier 1: ekspert 1.00 -> prog 0.70, wiec te same 0.30 to za malo
    st2 = _State(1)
    c2 = _make(tmp_path, st2, _Holdout(), "b")
    _tick(c2, 20, _Species(), 0.30)
    _tick(c2, 30, _Species(), 0.30)
    assert st2.current_tier == 1


def test_demotion_immediate_promotion_gradual(tmp_path):
    st, sp = _State(3), _Species()
    c = _make(tmp_path, st, _Holdout())
    _tick(c, 20, sp, 0.10)               # tier 3: prog degradacji 0.175
    _tick(c, 30, sp, 0.10)
    assert st.current_tier == 2
    assert st.resets == 1                # natychmiastowa wymiana map

    _tick(c, 100, sp, 1.0)
    _tick(c, 110, sp, 1.0)
    assert st.current_tier == 3
    assert st.resets == 1                # awans NIE resetuje - rotacja przejmuje


def test_stagnation_counters_reset_on_switch(tmp_path):
    st, sp = _State(1), _Species()
    c = _make(tmp_path, st, _Holdout())
    _tick(c, 20, sp, 1.0)
    _tick(c, 30, sp, 1.0)
    assert st.current_tier == 2
    for s in sp.species.values():
        assert s.last_improved == 30     # #26
        assert s.fitness_history == []


def test_cooldown_blocks_promotion_after_demotion(tmp_path):
    st, sp = _State(3), _Species()
    c = _make(tmp_path, st, _Holdout())
    _tick(c, 20, sp, 0.10)
    _tick(c, 30, sp, 0.10)
    assert st.current_tier == 2          # cooldown do generacji 60

    _tick(c, 50, sp, 1.0)
    _tick(c, 55, sp, 1.0)
    assert st.current_tier == 2          # warunek spelniony, ale cooldown blokuje

    _tick(c, 65, sp, 1.0)
    assert st.current_tier == 3