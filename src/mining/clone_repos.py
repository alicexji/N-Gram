from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

from git import Repo, GitCommandError


@dataclass
class RepoSpec:
    url: str
    name: Optional[str] = None  # if None, infer from URL


def infer_name(url: str) -> str:
    # https://github.com/org/repo.git -> repo
    tail = url.rstrip("/").split("/")[-1]
    return tail[:-4] if tail.endswith(".git") else tail


def clone_repositories(repos: List[RepoSpec], clone_dir: str) -> None:
    """
    Clone the given repositories into clone_dir.
    Skips repos already cloned
    Prints progress and errors without crashing the whole run
    """
    os.makedirs(clone_dir, exist_ok=True)

    for spec in repos:
        name = spec.name or infer_name(spec.url)
        dest = os.path.join(clone_dir, name)

        if os.path.exists(dest) and os.path.isdir(dest) and os.listdir(dest):
            print(f"[skip] {name} already exists at {dest}")
            continue

        print(f"[clone] {name} <- {spec.url}")
        try:
            Repo.clone_from(spec.url, dest)
        except GitCommandError as e:
            print(f"[error] failed to clone {name}: {e}")


def default_repo_list() -> List[RepoSpec]:
    """
    Starter list of reputable Java-heavy repos
    """
    return [
        RepoSpec("https://github.com/google/guava.git"),
        RepoSpec("https://github.com/apache/commons-lang.git"),
        RepoSpec("https://github.com/apache/commons-io.git"),
        RepoSpec("https://github.com/junit-team/junit5.git"),
        RepoSpec("https://github.com/square/okhttp.git"),
    ]
