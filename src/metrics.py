import re
import unicodedata
from typing import Optional
import editdistance



def normalize_text(text: str) -> str:

    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = re.sub(r"[^\w\s\u0900-\u097F]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── WER ───────────────────────────────────────────────────────────────────────

def word_error_rate(reference: str, hypothesis: str) -> float:

    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()

    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0

    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,       # deletion
                d[i][j - 1] + 1,       # insertion
                d[i - 1][j - 1] + cost # substitution
            )

    return d[len(ref_words)][len(hyp_words)] / len(ref_words)



def char_error_rate(reference: str, hypothesis: str) -> float:

    ref_chars = list(normalize_text(reference).replace(" ", ""))
    hyp_chars = list(normalize_text(hypothesis).replace(" ", ""))

    if len(ref_chars) == 0:
        return 0.0 if len(hyp_chars) == 0 else 1.0

    dist = editdistance.eval(ref_chars, hyp_chars)
    return dist / len(ref_chars)



def entity_accuracy(locality: str, hypothesis: str, fuzzy_threshold: float = 0.8) -> dict:

    loc_norm = normalize_text(locality)
    hyp_norm = normalize_text(hypothesis)

    exact = loc_norm in hyp_norm

    loc_chars = list(loc_norm.replace(" ", ""))
    hyp_chars = list(hyp_norm.replace(" ", ""))

    if not loc_chars:
        return {"exact": True, "fuzzy": True, "similarity": 1.0}

    best_sim = 0.0
    loc_len = len(loc_chars)

    if len(hyp_chars) >= loc_len:
        for start in range(len(hyp_chars) - loc_len + 1):
            window = hyp_chars[start: start + loc_len]
            dist = editdistance.eval(loc_chars, window)
            sim = 1 - dist / loc_len
            if sim > best_sim:
                best_sim = sim
    else:
        dist = editdistance.eval(loc_chars, hyp_chars)
        best_sim = 1 - dist / max(len(loc_chars), len(hyp_chars))

    fuzzy = best_sim >= fuzzy_threshold

    return {
        "exact": exact,
        "fuzzy": fuzzy,
        "similarity": round(best_sim, 4),
    }



def compute_all_metrics(
    reference: str,
    hypothesis: str,
    locality: str,
) -> dict:

    wer = word_error_rate(reference, hypothesis)
    cer = char_error_rate(reference, hypothesis)
    ear = entity_accuracy(locality, hypothesis)

    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)
    max_len = max(len(ref_norm), len(hyp_norm), 1)
    ned = editdistance.eval(ref_norm, hyp_norm) / max_len

    return {
        "wer": round(wer, 4),
        "cer": round(cer, 4),
        "entity_exact": ear["exact"],
        "entity_fuzzy": ear["fuzzy"],
        "entity_similarity": ear["similarity"],
        "normalized_edit_distance": round(ned, 4),
    }


def aggregate_results(results: list[dict]) -> dict:

    if not results:
        return {}

    n = len(results)
    metrics = ["wer", "cer", "entity_similarity", "normalized_edit_distance"]
    bool_metrics = ["entity_exact", "entity_fuzzy"]

    agg = {}
    for m in metrics:
        vals = [r.get(m, 0) for r in results if r.get(m) is not None]
        agg[f"mean_{m}"] = round(sum(vals) / len(vals), 4) if vals else None

    for m in bool_metrics:
        vals = [r.get(m, False) for r in results]
        agg[f"accuracy_{m}"] = round(sum(vals) / n, 4)

    agg["n_samples"] = n
    return agg
