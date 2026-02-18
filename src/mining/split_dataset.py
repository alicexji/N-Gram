from __future__ import annotations
import os
import random
from typing import List


def read_lines(path: str) -> List[str]:
    with open(path, "rb") as f:
        lines = f.read().decode("utf-8", errors="ignore").splitlines()
    # drop empty lines
    return [ln.strip() for ln in lines if ln.strip()]


def write_lines(lines: List[str], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        for ln in lines:
            f.write((ln + "\n").encode("utf-8", errors="ignore"))


def make_splits(
    all_methods_path: str,
    out_dir: str,
    seed: int = 42,
    t1_max: int = 15000,
    t2_max: int = 25000,
    t3_max: int = 35000,
    val_size: int = 1000,
    test_size: int = 1000,
) -> None:
    methods = read_lines(all_methods_path)
    random.Random(seed).shuffle(methods)

    needed = t3_max + val_size + test_size
    if len(methods) < needed:
        raise ValueError(f"Need at least {needed} methods, but only have {len(methods)}.")

    T1 = methods[:t1_max]
    T2 = methods[:t2_max]
    T3 = methods[:t3_max]
    V  = methods[t3_max : t3_max + val_size]
    Te = methods[t3_max + val_size : t3_max + val_size + test_size]

    write_lines(T1, os.path.join(out_dir, "T1.txt"))
    write_lines(T2, os.path.join(out_dir, "T2.txt"))
    write_lines(T3, os.path.join(out_dir, "T3.txt"))
    write_lines(V,  os.path.join(out_dir, "val.txt"))
    write_lines(Te, os.path.join(out_dir, "test_self.txt"))

    print("Split sizes:")
    print(f"  T1: {len(T1)}")
    print(f"  T2: {len(T2)}")
    print(f"  T3: {len(T3)}")
    print(f"  V : {len(V)}")
    print(f"  Te: {len(Te)}")
