import argparse

from src.mining.github_api import fetch_top_java_repos
from src.mining.clone_repos import RepoSpec, clone_repositories
from src.mining.extract_methods import mine_all_methods, save_methods
from src.mining.split_dataset import make_splits


CLONE_DIR = "data/raw/repos"
ALL_METHODS = "data/processed/all_methods.txt"
PROCESSED_DIR = "data/processed"


def safe_folder_name(full_name: str) -> str:
    return full_name.replace("/", "__")


def stage_mine(max_methods: int = 70000):
    # GitHub mining settings
    MIN_STARS = 1000
    MIN_SIZE_KB = 1000
    PUSHED_AFTER = "2024-02-17"
    NUM_REPOS = 40

    print("Fetching filtered Java repos from GitHub Search API ...")
    api_repos = fetch_top_java_repos(
        num_repos=NUM_REPOS,
        min_stars=MIN_STARS,
        min_size_kb=MIN_SIZE_KB,
        pushed_after=PUSHED_AFTER,
        per_page=100,
    )
    print(f"Fetched {len(api_repos)} repos")

    seed = [RepoSpec("https://github.com/google/guava.git", name="google__guava")]

    api_specs = [
        RepoSpec(r.clone_url, name=safe_folder_name(r.full_name))
        for r in api_repos
        if r.full_name.lower() != "google/guava"
    ]

    repos_to_clone = seed + api_specs

    print(f"\nCloning {len(repos_to_clone)} repos into {CLONE_DIR} ...")
    clone_repositories(repos_to_clone, CLONE_DIR)
    print("Cloning complete.\n")

    print(f"Mining methods (max_methods={max_methods}) ...")
    methods = mine_all_methods(CLONE_DIR, max_methods=max_methods)
    save_methods(methods, ALL_METHODS)


def stage_split():
    make_splits(
        all_methods_path=ALL_METHODS,
        out_dir=PROCESSED_DIR,
        seed=42
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["mine", "split", "all"], default="all")
    parser.add_argument("--max_methods", type=int, default=70000)
    args = parser.parse_args()

    if args.stage in ("mine", "all"):
        stage_mine(max_methods=args.max_methods)

    if args.stage in ("split", "all"):
        stage_split()


if __name__ == "__main__":
    main()
