import argparse

from src.mining.github_api import fetch_top_java_repos
from src.mining.clone_repos import RepoSpec, clone_repositories
from src.mining.extract_methods import mine_all_methods, save_methods
from src.mining.split_dataset import make_splits
from src.modeling.backoff_ngram_model import BackoffNGramModel


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



# def stage_train():
#     from src.modeling.train_validate import TrainConfig, train_and_validate

#     train_sets = ["T1", "T2", "T3"]
#     all_results = {}

#     for ts in train_sets:   # loop training n = [3,5,7] for each training set
#         print("\n" + "=" * 60)
#         print(f"TRAINING SET: {ts}")
#         print("=" * 60)

#         cfg = TrainConfig(
#             train_path=f"data/processed/{ts}.txt",
#             val_path="data/processed/val.txt",
#             n_values=[3, 5, 7],
#             alpha=0.1
#         )
#         best_n, results = train_and_validate(cfg)
#         all_results[ts] = (best_n, results)

#     print("\nSummary (validation perplexity):")
#     for ts in train_sets:
#         best_n, results = all_results[ts]
#         r3 = results.get(3)
#         r5 = results.get(5)
#         r7 = results.get(7)
#         print(f"{ts}: n=3 {r3:.4f} | n=5 {r5:.4f} | n=7 {r7:.4f} | best={best_n}")

# def run_backoff(train_path: str, val_path: str, n_values=(3,5,7), beta=0.4, unigram_alpha=0.1):
#     results = {}
#     for n in n_values:
#         print(f"\nBackoff {n}-gram (beta={beta}, unigram_alpha={unigram_alpha}) on {train_path} ...")
#         m = BackoffNGramModel(n=n, beta=beta, unigram_alpha=unigram_alpha)
#         m.train_from_file(train_path)
#         pp = m.perplexity(val_path)
#         results[n] = pp
#         print(f"{n}-gram backoff validation perplexity: {pp:.4f}")
#     best_n = min(results, key=results.get)
#     print(f"\nBest backoff n: {best_n} (PP={results[best_n]:.4f})")
#     return results

def stage_train():
    """
    train using both add-alpha AND backoff and see which one is better. 
    Backoff will probably be better for higher order ngrams.
    """
    # Add-alpha baseline
    from src.modeling.train_validate import TrainConfig, train_and_validate

    # Backoff alternative
    from src.modeling.backoff_ngram_model import BackoffNGramModel

    train_sets = ["T1", "T2", "T3"]
    val_path = "data/processed/val.txt"

    # Add-alpha results
    addalpha_summary = {}

    for ts in train_sets:
        print("\n" + "=" * 60)
        print(f"[ADD-ALPHA] TRAINING SET: {ts}")
        print("=" * 60)

        cfg = TrainConfig(
            train_path=f"data/processed/{ts}.txt",
            val_path=val_path,
            n_values=[3, 5, 7],
            alpha=0.1
        )
        best_n, results = train_and_validate(cfg)
        addalpha_summary[ts] = (best_n, results)

    print("\nAdd-alpha Summary (validation perplexity):")
    for ts in train_sets:
        best_n, results = addalpha_summary[ts]
        print(
            f"{ts}: n=3 {results[3]:.4f} | n=5 {results[5]:.4f} | n=7 {results[7]:.4f} | best={best_n}"
        )

    # Backoff results
    BETA = 0.4
    UNIGRAM_ALPHA = 0.1     # to be used when we back up all the way to unigram

    backoff_summary = {}

    for ts in train_sets:
        train_path = f"data/processed/{ts}.txt"

        print("\n" + "=" * 60)
        print(f"[BACKOFF] TRAINING SET: {ts} (beta={BETA}, unigram_alpha={UNIGRAM_ALPHA})")
        print("=" * 60)

        results = {}
        for n in [3, 5, 7]:
            print(f"\nTraining backoff {n}-gram on {train_path} ...")
            m = BackoffNGramModel(n=n, beta=BETA, unigram_alpha=UNIGRAM_ALPHA)
            m.train_from_file(train_path)
            pp = m.perplexity(val_path)
            results[n] = pp
            print(f"backoff {n}-gram validation perplexity: {pp:.4f}")

        best_n = min(results, key=results.get)
        print(f"\nBest backoff n on validation: {best_n} (PP={results[best_n]:.4f})")
        backoff_summary[ts] = (best_n, results)

    print("\nBackoff Summary (validation perplexity):")
    for ts in train_sets:
        best_n, results = backoff_summary[ts]
        print(
            f"{ts}: n=3 {results[3]:.4f} | n=5 {results[5]:.4f} | n=7 {results[7]:.4f} | best={best_n}"
        )



def stage_json():
    from src.modeling.ngram_model import NGramModel
    from src.evaluation.json_output import evaluate_to_json
    import os

    # Best config from validation
    BEST_N = 3
    ALPHA = 0.1
    TRAIN_PATH = "data/processed/T3.txt"

    model = NGramModel(n=BEST_N, alpha=ALPHA)
    model.train_from_file(TRAIN_PATH)

    os.makedirs("results", exist_ok=True)

    # Self-created test set
    evaluate_to_json(
        model=model,
        test_path="data/processed/test_self.txt",
        out_path="results/results-self.json",
        context_window=BEST_N,
        testset_name="test_self.txt"
    )

    # Provided test set
    provided_path = "data/processed/provided.txt"
    if os.path.exists(provided_path):       # if test set hasn't been provided yet, we skip
        evaluate_to_json(
            model=model,
            test_path=provided_path,
            out_path="results/results-provided.json",
            context_window=BEST_N,
            testset_name="provided.txt"
        )
    else:
        print("NOTE: data/processed/provided.txt not found yet. Add it, then rerun --stage json.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["mine", "split", "train", "json", "all"], default="all")
    parser.add_argument("--max_methods", type=int, default=70000)
    args = parser.parse_args()

    if args.stage in ("mine", "all"):
        stage_mine(max_methods=args.max_methods)

    if args.stage in ("split", "all"):
        stage_split()

    if args.stage in ("train", "all"):
        stage_train()

    if args.stage in ("json", "all"):
        stage_json()



if __name__ == "__main__":
    main()
