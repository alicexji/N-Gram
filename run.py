from src.mining.clone_repos import clone_repositories, default_repo_list


CLONE_DIR = "data/raw/repos"


def main():

    print("Step 1: Cloning repositories...")

    repos = default_repo_list()

    clone_repositories(
        repos=repos,
        clone_dir=CLONE_DIR
    )

    print("Cloning complete.")


if __name__ == "__main__":
    main()
