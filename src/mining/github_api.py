from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

import requests


@dataclass
class RepoInfo:
    full_name: str
    clone_url: str
    stars: int
    description: str = ""
    updated_at: str = ""


def fetch_top_java_repos(
    num_repos: int = 200,
    per_page: int = 100,
    min_stars: int = 1000,
    min_size_kb: int = 1000,          # ~1MB repo size floor
    pushed_after: str = "2024-02-17",  # a year ago from today
    sleep_s: float = 0.2,
) -> List[RepoInfo]:
    repos: List[RepoInfo] = []
    page = 1
    per_page = min(per_page, 100)

    while len(repos) < num_repos and page <= 10:
        url = "https://api.github.com/search/repositories"
        q = (
            f"language:java stars:>={min_stars} "
            f"fork:false size:>={min_size_kb} pushed:>={pushed_after}"
        )
        params = {
            "q": q,
            "sort": "stars",
            "order": "desc",
            "per_page": per_page,
            "page": page,
        }

        resp = requests.get(url, params=params, timeout=30)

        if resp.status_code != 200:
            print(f"GitHub API error: {resp.status_code}")
            try:
                print(resp.json())
            except Exception:
                print(resp.text)
            break

        data = resp.json()
        items = data.get("items", [])
        if not items:
            break

        for item in items:
            repos.append(
                RepoInfo(
                    full_name=item["full_name"],
                    clone_url=item["clone_url"],
                    stars=item["stargazers_count"],
                    description=item.get("description") or "",
                    updated_at=item.get("updated_at") or "",
                )
            )
            if len(repos) >= num_repos:
                break

        page += 1
        time.sleep(sleep_s)

    return repos[:num_repos]


def repo_has_min_commits(full_name: str, min_commits: int = 100) -> bool:
    """
    Approximate commit count using GitHub commits API.
    Note: this method is not used because we aren't actually able to retrieve the # of commits by looking at pagination
    """
    url = f"https://api.github.com/repos/{full_name}/commits"
    params = {"per_page": 1}

    try:
        resp = requests.get(url, params=params, timeout=30)
    except Exception:
        return False

    if resp.status_code != 200:
        return False

    link = resp.headers.get("Link", "")

    # If Link header has rel="last", extract page number
    if 'rel="last"' in link:
        parts = link.split(",")
        last = [p for p in parts if 'rel="last"' in p]
        if last:
            segment = last[0]
            idx = segment.find("page=")
            if idx != -1:
                idx += 5
                num = ""
                while idx < len(segment) and segment[idx].isdigit():
                    num += segment[idx]
                    idx += 1
                if num:
                    return int(num) >= min_commits

    # If no pages, repo has very few commits
    return False
