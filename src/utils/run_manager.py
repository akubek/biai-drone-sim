import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.core.environment import TIERS


def _git_info() -> dict:
    """Commit hash and information about whether the working tree had uncommitted changes."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
        ).strip())
        return {"git_commit": commit, "git_dirty": dirty}
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return {"git_commit": "unknown", "git_dirty": None}


def create_run_dir(arch: str, exp_config: dict, config_path: str) -> Path:
    """Creates a run directory, copies configuration files, and writes the initial manifest."""
    seed = exp_config.get("seed", "noseed")
    mode = exp_config.get("training_mode", 0)
    stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Przebiegi z nadpisana populacja traktujemy jako testy smoke - oznaczamy, zeby nie mylic ich z wynikami.
    # potencjalnie do zmiany
    prefix = "smoke_" if exp_config.get("pop_size") is not None else ""
    run_dir = Path("results") / f"{prefix}{stamp}_{arch}_mode{mode}_seed{seed}"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    # Kopie konfiguracji - treść (mogą się zmieniać)
    shutil.copy(config_path, run_dir / "conf_neat.txt")
    shutil.copy(Path("src") / "training_config.json", run_dir / "training_config.json")
    shutil.copy(Path("conf") / "curriculum.json", run_dir / "curriculum.json")
    shutil.copy(Path("conf") / "baselines.json", run_dir / "baselines.json")
    for name in ("rewards.py", "evolution.py", "physics.py", "config.py"):
        shutil.copy(Path("src") / "config" / name, run_dir / f"config_{name}")

    manifest = {
        "timestamp_start": datetime.now(tz=timezone.utc).isoformat(),
        "arch": arch,
        "python": sys.version.split()[0],
        "exp_config": exp_config,
        "status": "running",
        **_git_info(),
        "tiers": TIERS,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return run_dir


def finalize_run(run_dir: Path, **results) -> None:
    """Writes the final results and status to the manifest."""
    path = run_dir / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(results)
    manifest["timestamp_end"] = datetime.now(tz=timezone.utc).isoformat()
    manifest["status"] = "finished"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")