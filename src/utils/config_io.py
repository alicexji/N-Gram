from __future__ import annotations
import json
import os
from typing import Any, Dict


def save_best_config(cfg: Dict[str, Any], path: str = "results/best_config.json") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"\nSaved best config -> {path}")


def load_best_config(path: str = "results/best_config.json") -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)