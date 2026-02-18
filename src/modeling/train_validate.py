from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .ngram_model import NGramModel


@dataclass
class TrainConfig:
    train_path: str
    val_path: str
    n_values: List[int] = None
    alpha: float = 0.1


def train_and_validate(cfg: TrainConfig) -> Tuple[int, Dict[int, float]]:
    if cfg.n_values is None:
        cfg.n_values = [3, 5, 7]

    results: Dict[int, float] = {}

    for n in cfg.n_values:
        print(f"\nTraining {n}-gram (alpha={cfg.alpha}) on {cfg.train_path} ...")
        model = NGramModel(n=n, alpha=cfg.alpha)
        model.train_from_file(cfg.train_path)

        pp = model.perplexity(cfg.val_path)
        results[n] = pp
        print(f"{n}-gram validation perplexity: {pp:.4f}")

    best_n = min(results, key=results.get)
    print(f"\nBest n on validation: {best_n} (PP={results[best_n]:.4f})")
    return best_n, results
