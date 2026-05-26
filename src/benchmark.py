import os
import sys
import json
import time
import csv
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics import compute_all_metrics, aggregate_results


def get_model(model_id: str):
    model_id = model_id.strip().lower()

    if model_id == "deepgram":
        from src.models.deepgram_asr import DeepgramASR
        return DeepgramASR()

    elif model_id in ("groq_whisper", "groq"):
        from src.models.groq_whisper_asr import GroqWhisperASR
        return GroqWhisperASR()

    elif model_id == "assemblyai":
        from src.models.assemblyai_asr import AssemblyAIASR
        return AssemblyAIASR()

    elif model_id == "sarvam":
        from src.models.sarvam_asr import SarvamASR
        return SarvamASR()

    else:
        raise ValueError(f"Unknown model: {model_id}. "
                         f"Available: deepgram, groq_whisper, assemblyai, sarvam")



def load_ground_truth(audio_dir: Path) -> list[dict]:
    gt_path = audio_dir / "ground_truth.csv"
    if not gt_path.exists():
        raise FileNotFoundError(
            f"ground_truth.csv not found in {audio_dir}. "
            "Run: python scripts/generate_audio.py"
        )

    samples = []
    with open(gt_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filepath = Path(row["filepath"])
            if not filepath.exists():
                # Try relative path
                filepath = audio_dir / row["filename"]
            if filepath.exists():
                samples.append({
                    **row,
                    "filepath": str(filepath),
                })
            else:
                print(f"  [WARN] Audio file not found: {row['filename']} — skipping")

    return samples



def run_model(model_id: str, samples: list[dict], verbose: bool = True) -> list[dict]:

    print(f"\n{'─'*60}")
    print(f"  Running model: {model_id.upper()}")
    print(f"{'─'*60}")

    try:
        model = get_model(model_id)
    except (ValueError, ImportError) as e:
        print(f"  [ERROR] Could not load model '{model_id}': {e}")
        return []

    results = []
    for i, sample in enumerate(samples):
        if verbose:
            print(f"  [{i+1:02d}/{len(samples)}] {sample['filename']}")

        try:
            asr_result = model.transcribe_file(sample["filepath"])
        except Exception as e:
            asr_result = {
                "transcript": "",
                "confidence": 0.0,
                "latency_ms": 0.0,
                "model": model_id,
                "error": str(e),
            }

        reference = sample["sentence"]
        hypothesis = asr_result["transcript"]
        locality = sample["locality"]

        metrics = compute_all_metrics(reference, hypothesis, locality)

        result = {
            "model": model_id,
            "index": sample["index"],
            "locality": locality,
            "condition": sample["condition"],
            "reference": reference,
            "hypothesis": hypothesis,
            "filename": sample["filename"],
            "confidence": asr_result.get("confidence", 0.0),
            "latency_ms": asr_result.get("latency_ms", 0.0),
            "error": asr_result.get("error"),
            **metrics,
        }

        if verbose:
            mark = "✓" if metrics["entity_exact"] else ("~" if metrics["entity_fuzzy"] else "✗")
            print(f"    {mark} Entity: {locality}")
            print(f"      REF: {reference}")
            print(f"      HYP: {hypothesis[:80]}")
            print(f"      WER={metrics['wer']:.2f} CER={metrics['cer']:.2f} "
                  f"EntitySim={metrics['entity_similarity']:.2f} "
                  f"Latency={asr_result.get('latency_ms',0):.0f}ms")

        results.append(result)

    return results



def print_comparison_table(all_results: dict[str, list[dict]]):
    try:
        from tabulate import tabulate
        use_tabulate = True
    except ImportError:
        use_tabulate = False

    header = ["Model", "WER↓", "CER↓", "Exact%", "Fuzzy%", "EntitySim↑", "Latency(ms)↓", "Errors"]
    rows = []

    for model_id, results in all_results.items():
        if not results:
            rows.append([model_id, "N/A"] * (len(header) - 1))
            continue

        agg = aggregate_results(results)
        valid = [r for r in results if not r.get("error")]
        err_count = len(results) - len(valid)
        avg_lat = (sum(r["latency_ms"] for r in valid) / len(valid)) if valid else 0

        rows.append([
            model_id,
            f"{agg.get('mean_wer', 0):.3f}",
            f"{agg.get('mean_cer', 0):.3f}",
            f"{agg.get('accuracy_entity_exact', 0)*100:.1f}%",
            f"{agg.get('accuracy_entity_fuzzy', 0)*100:.1f}%",
            f"{agg.get('mean_entity_similarity', 0):.3f}",
            f"{avg_lat:.0f}",
            str(err_count),
        ])

    print(f"\n{'='*70}")
    print("  ASR BENCHMARKING RESULTS")
    print(f"{'='*70}")

    if use_tabulate:
        print(tabulate(rows, headers=header, tablefmt="rounded_outline"))
    else:
        print("  " + " | ".join(header))
        print("  " + "-" * 60)
        for row in rows:
            print("  " + " | ".join(str(c) for c in row))

    print()



def print_failure_analysis(all_results: dict[str, list[dict]]):
    print(f"\n{'='*70}")
    print("  FAILURE ANALYSIS")
    print(f"{'='*70}")

    conditions = set()
    for results in all_results.values():
        for r in results:
            conditions.add(r["condition"])

    print("\n  [By Recording Condition]")
    header = ["Condition"] + list(all_results.keys())
    rows = []
    for cond in sorted(conditions):
        row = [cond]
        for model_id, results in all_results.items():
            cond_results = [r for r in results if r["condition"] == cond]
            if cond_results:
                ear = sum(r["entity_exact"] for r in cond_results) / len(cond_results)
                row.append(f"{ear*100:.0f}%")
            else:
                row.append("—")
        rows.append(row)

    try:
        from tabulate import tabulate
        print(tabulate(rows, headers=header, tablefmt="simple"))
    except ImportError:
        for row in rows:
            print("  " + " | ".join(str(c) for c in row))

    print("\n  [Hardest Localities — lowest avg entity similarity across models]")
    locality_scores = {}
    for results in all_results.values():
        for r in results:
            loc = r["locality"]
            if loc not in locality_scores:
                locality_scores[loc] = []
            locality_scores[loc].append(r["entity_similarity"])

    avg_scores = {loc: sum(v)/len(v) for loc, v in locality_scores.items()}
    hardest = sorted(avg_scores.items(), key=lambda x: x[1])[:5]

    for loc, score in hardest:
        print(f"    {loc:<25} avg_similarity={score:.3f}")



def save_results(all_results: dict[str, list[dict]], results_dir: Path):
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Raw JSON
    raw_path = results_dir / f"raw_results_{timestamp}.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)

    all_rows = []
    for model_id, results in all_results.items():
        all_rows.extend(results)

    if all_rows:
        csv_path = results_dir / f"per_sample_{timestamp}.csv"
        fieldnames = [k for k in all_rows[0].keys() if k != "raw"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rows)

    summary_rows = []
    for model_id, results in all_results.items():
        if results:
            agg = aggregate_results(results)
            valid = [r for r in results if not r.get("error")]
            avg_lat = sum(r["latency_ms"] for r in valid) / len(valid) if valid else 0
            summary_rows.append({
                "model": model_id,
                "n_samples": len(results),
                "n_errors": len(results) - len(valid),
                "avg_latency_ms": round(avg_lat, 2),
                **agg,
            })

    if summary_rows:
        sum_path = results_dir / f"summary_{timestamp}.csv"
        with open(sum_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)

    print(f"\n  Results saved to: {results_dir}")
    print(f"    Raw JSON  : {raw_path.name}")
    if all_rows:
        print(f"    Per-sample: {csv_path.name}")
    if summary_rows:
        print(f"    Summary   : {sum_path.name}")

    return raw_path



def main():
    parser = argparse.ArgumentParser(
        description="ASR Shootout Benchmark Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/benchmark.py
  python src/benchmark.py --models deepgram groq_whisper
  python src/benchmark.py --audio_dir ./my_audio --no_verbose
        """
    )
    parser.add_argument(
        "--models", nargs="+",
        default=None,
        help="Models to run (default: from MODELS_TO_RUN in .env)"
    )
    parser.add_argument(
        "--audio_dir", type=str,
        default=None,
        help="Directory with audio files and ground_truth.csv"
    )
    parser.add_argument(
        "--results_dir", type=str,
        default=None,
        help="Output directory for results"
    )
    parser.add_argument(
        "--no_verbose", action="store_true",
        help="Suppress per-sample output"
    )
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir or os.getenv("AUDIO_DIR", "./audio_samples"))
    results_dir = Path(args.results_dir or os.getenv("RESULTS_DIR", "./results"))

    if args.models:
        model_ids = args.models
    else:
        env_models = os.getenv("MODELS_TO_RUN", "deepgram,groq_whisper")
        model_ids = [m.strip() for m in env_models.split(",") if m.strip()]

    verbose = not args.no_verbose

    print(f"\n{'='*60}")
    print("  ASR SHOOTOUT — Benchmark Pipeline")
    print(f"{'='*60}")
    print(f"  Audio dir  : {audio_dir.resolve()}")
    print(f"  Results dir: {results_dir.resolve()}")
    print(f"  Models     : {', '.join(model_ids)}")
    print(f"{'='*60}")

    try:
        samples = load_ground_truth(audio_dir)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

    print(f"\n  Loaded {len(samples)} audio samples.")

    all_results = {}
    for model_id in model_ids:
        results = run_model(model_id, samples, verbose=verbose)
        all_results[model_id] = results

    print_comparison_table(all_results)
    print_failure_analysis(all_results)

    save_results(all_results, results_dir)

    print("\n  Done! Run python src/visualize.py to generate charts.\n")


if __name__ == "__main__":
    main()
