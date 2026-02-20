from __future__ import annotations
from collections import defaultdict

import json
import math
from typing import Dict, List, Tuple

from src.modeling.ngram_model import NGramModel


def _read_lines(path: str) -> List[List[str]]:
    methods = []
    with open(path, "rb") as f:
        for line in f:
            toks = line.decode("utf-8", errors="ignore").strip().split()
            if toks:
                methods.append(toks)
    return methods

# def build_next_token_index(model: NGramModel):
#     """
#     Build a map: context(tuple) -> dict(next_token -> count)
#     This makes argmax prediction fast.
#     """
#     next_map = defaultdict(lambda: defaultdict(int))
#     for ngram, cnt in model.ngram_counts.items():
#         ctx = ngram[:-1]
#         nxt = ngram[-1]
#         next_map[ctx][nxt] += cnt
#     return next_map

def build_next_token_index(model):
    """
    Build a map: context(tuple) -> dict(next_token -> count)
    Works for both NGramModel and BackoffNGramModel
    """

    next_map = defaultdict(lambda: defaultdict(int))

    # Case 1: Add-alpha model
    if hasattr(model, "ngram_counts"):
        source = model.ngram_counts

    # Case 2: Backoff model
    elif hasattr(model, "counts_by_order"):
        source = model.counts_by_order[model.n]

    else:
        raise ValueError("Unknown model type for next-token index")

    for ngram, cnt in source.items():
        ctx = ngram[:-1]
        nxt = ngram[-1]
        next_map[ctx][nxt] += cnt

    return next_map

# def _predict_next_token(model: NGramModel, context: List[str]) -> Tuple[str, float]:
#     """
#     Returns (argmax_token, argmax_probability) under P(. | context).
#     """
#     best_tok = None
#     best_p = -1.0
#     for tok in model.vocab:
#         p = model.prob(context, tok)
#         if p > best_p:
#             best_p = p
#             best_tok = tok
#     return best_tok, best_p

def _predict_next_token(model: NGramModel, context: List[str], next_map) -> Tuple[str, float]:
    ctx = tuple([t if t in model.vocab else model.UNK for t in context])

    candidates = next_map.get(ctx)
    if not candidates:
        p = model.prob(list(ctx), model.UNK)
        return model.UNK, float(p)

    # Pick the most frequent next token for this context (fast argmax approximation)
    best_tok = max(candidates.items(), key=lambda kv: kv[1])[0]
    best_p = model.prob(list(ctx), best_tok)
    return best_tok, float(best_p)



def evaluate_to_json(
    model: NGramModel,
    test_path: str,
    out_path: str,
    context_window: int,
    testset_name: str = None,
    max_positions_per_method: int = None,
) -> Dict:
    """
    Creates the JSON structure
    overall perplexity on the test set (ground-truth probabilities)
    per-method list of per-position predictions with context, predToken, predProbability, groundTruth
    """
    methods = _read_lines(test_path)
    data = []
    next_map = build_next_token_index(model)

    total_log_sum = 0.0
    total_N = 0

    total_methods = len(methods)

    for idx, toks in enumerate(methods, start=1):

        if idx % 10 == 0:
            print(f"JSON progress: {idx}/{len(methods)} methods...")    #print progress every 10 methods

            
        # map OOV to UNK using model vocab
        mapped = [t if t in model.vocab else model.UNK for t in toks]
        padded = [model.BOS] * (context_window - 1) + mapped + [model.EOS]

        preds = []
        positions_done = 0

        for i in range(context_window - 1, len(padded)):
            context = padded[i - (context_window - 1): i]
            gt = padded[i]

            # ground-truth probability for perplexity (required) :contentReference[oaicite:2]{index=2}
            p_gt = model.prob(context, gt)
            total_log_sum += math.log(p_gt)
            total_N += 1

            # argmax prediction for reporting
            pred_tok, pred_p = _predict_next_token(model, context, next_map)

            preds.append({
                "context": context,
                "predToken": pred_tok,
                "predProbability": float(pred_p),
                "groundTruth": gt
            })

            positions_done += 1
            if max_positions_per_method is not None and positions_done >= max_positions_per_method:
                break

        data.append({
            "index": f"ID{idx}",
            "tokenizedCode": " ".join(toks),
            "contextWindow": context_window,
            "predictions": preds
        })

    # total_n > 0 ensures we don't divide by 0, so we return infinity
    perplexity = math.exp(-total_log_sum / total_N) if total_N > 0 else float("inf")   

    payload = {
        "testSet": testset_name or test_path,
        "perplexity": float(perplexity),
        "data": data
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote: {out_path}")
    print(f"Test perplexity: {perplexity:.4f}")

    return payload
