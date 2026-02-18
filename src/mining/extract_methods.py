import os
from pathlib import Path
from typing import List, Set

import javalang
from javalang.tokenizer import tokenize


MIN_TOKENS = 10


def clean_non_ascii(text: str) -> str:
    return text.encode("ascii", errors="ignore").decode()


def tokenize_method(method_code: str) -> List[str]:
    try:
        tokens = list(tokenize(method_code))
        return [t.value for t in tokens]
    except:
        return []


def extract_methods_from_file(java_file: Path) -> List[str]:
    methods = []

    try:
        with open(java_file, "r", encoding="utf-8", errors="ignore") as f:
            code = f.read()

        code = clean_non_ascii(code)

        tree = javalang.parse.parse(code)

        lines = code.split("\n")

        for _, node in tree.filter(javalang.tree.MethodDeclaration):

            if not node.position:
                continue

            start_line = node.position.line - 1

            method_code = "\n".join(lines[start_line:])

            tokens = tokenize_method(method_code)

            if len(tokens) < MIN_TOKENS:
                continue

            tokenized_line = " ".join(tokens)

            methods.append(tokenized_line)

    except:
        pass

    return methods


def mine_all_methods(repo_dir: str, max_methods: int = 70000) -> List[str]:

    repo_path = Path(repo_dir)

    all_methods: List[str] = []
    seen: Set[str] = set()

    java_files = list(repo_path.rglob("*.java"))

    print(f"Found {len(java_files)} Java files")

    for java_file in java_files:

        methods = extract_methods_from_file(java_file)

        for method in methods:

            if method not in seen:      #avoid duplicates
                seen.add(method)
                all_methods.append(method)
                if len(all_methods) >= max_methods:
                    print(f"Reached max_methods={max_methods}. Stopping early.")
                    return all_methods


    print(f"Extracted {len(all_methods)} unique methods")

    return all_methods


def save_methods(methods, output_file: str):
    import os
    import re

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Surrogates are in the range U+D800 to U+DFFF and cannot be UTF-8 encoded
    surrogate_re = re.compile(r"[\ud800-\udfff]")

    with open(output_file, "wb") as f:  # write raw bytes
        for method in methods:
            cleaned = surrogate_re.sub("", method)
            line = (cleaned + "\n").encode("utf-8", errors="ignore")
            f.write(line)

    print(f"Saved methods to {output_file}")
