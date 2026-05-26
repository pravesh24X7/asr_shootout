import os
import sys
import json
import random
import csv
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics import compute_all_metrics, aggregate_results

AUDIO_DIR = Path(os.getenv("AUDIO_DIR", "./audio_samples"))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "./results"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


LOCALITIES = [
    "Koramangala", "Indiranagar", "Whitefield", "Electronic City",
    "Marathahalli", "Jayanagar", "Rajajinagar", "Hebbal", "Yelahanka",
    "Banashankari", "HSR Layout", "BTM Layout", "Majestic", "Silk Board",
    "Bellandur", "Sarjapur", "Bommanahalli", "KR Puram", "Peenya", "Yeshwanthpur",
]

TEMPLATES = [
    "Haan, main {loc} mein rehta hoon",
    "Mera ghar {loc} ke paas hai",
    "Main {loc} se hoon",
    "Ji, {loc} area mein",
    "Sir, main {loc} wala hoon",
    "Mera address hai {loc}",
    "Main {loc} mein kaam karta hoon",
    "Haan bhai, {loc} hi rehta hoon",
]

CONDITIONS = ["quiet_room", "street_noise", "phone_call", "rushed", "whispered"]

random.seed(42)

SAMPLES = []
for i, loc in enumerate(LOCALITIES):
    SAMPLES.append({
        "index": i + 1,
        "locality": loc,
        "sentence": random.choice(TEMPLATES).format(loc=loc),
        "condition": CONDITIONS[i % len(CONDITIONS)],
        "filename": f"{i+1:02d}_{loc.lower().replace(' ','_')}_{CONDITIONS[i%len(CONDITIONS)]}.wav",
    })



def simulate_transcription(model: str, sentence: str, locality: str, condition: str) -> dict:

    accuracy_map = {
        "deepgram": {"quiet_room": 0.90, "street_noise": 0.70, "phone_call": 0.75,
                     "rushed": 0.65, "whispered": 0.50},
        "groq_whisper": {"quiet_room": 0.93, "street_noise": 0.75, "phone_call": 0.80,
                         "rushed": 0.72, "whispered": 0.60},
        "assemblyai": {"quiet_room": 0.88, "street_noise": 0.68, "phone_call": 0.72,
                       "rushed": 0.60, "whispered": 0.45},
        "sarvam": {"quiet_room": 0.92, "street_noise": 0.78, "phone_call": 0.82,
                   "rushed": 0.75, "whispered": 0.65},
    }

    acc = accuracy_map.get(model, {}).get(condition, 0.75)

    entity_correct = random.random() < acc

    substitutions = {
        "Koramangala": ["Koromangala", "Coramangala", "Koramangla"],
        "Marathahalli": ["Marathaally", "Marathhalli", "Maratahalli"],
        "Rajajinagar": ["Rajaji nagar", "Raju nagar", "Rajajinagara"],
        "Banashankari": ["Banashankri", "Banashankari", "Bansankari"],
        "Yeshwanthpur": ["Yashwantpur", "Yeshwantpur", "Yeshwanthpura"],
        "Bommanahalli": ["Bomanahali", "Bomana halli", "Bomanahalli"],
        "Doddanekundi": ["Dodda nekundi", "Doddanekunde", "Dodanekundi"],
        "Kengeri Upanagara": ["Kengeri", "Kangeri Upanagara", "Kengeri upnagara"],
        "Thalaghattapura": ["Thalagatapura", "Talaghatapura", "Thalaghatta pura"],
        "Kadugondanahalli": ["Kadugondana halli", "Kadugondnahalli", "Kadugondanahali"],
        "Hesaraghatta": ["Hesaragata", "Hesaraghata", "Hessaraghatta"],
        "Byatarayanapura": ["Byatarayana pura", "Byataranapura", "Batarayanapura"],
    }

    words = sentence.split()
    if entity_correct:
        hypothesis = sentence
        if random.random() < 0.3:
            hypothesis = sentence.lower()
    else:
        error_options = substitutions.get(locality, [locality + "a", locality[:-1]])
        wrong_loc = random.choice(error_options)
        hypothesis = sentence.replace(locality, wrong_loc)
        if condition in ("street_noise", "whispered") and random.random() < 0.4:
            words_h = hypothesis.split()
            if len(words_h) > 2:
                drop_idx = random.randint(0, len(words_h) - 1)
                words_h.pop(drop_idx)
                hypothesis = " ".join(words_h)

    base_latency = {
        "deepgram": 450, "groq_whisper": 850, "assemblyai": 3500, "sarvam": 600
    }
    latency_ms = base_latency.get(model, 1000) + random.gauss(0, 80)
    latency_ms = max(200, latency_ms)

    return {
        "transcript": hypothesis,
        "confidence": round(acc + random.gauss(0, 0.05), 3),
        "latency_ms": round(latency_ms, 1),
        "model": model,
        "error": None,
    }



def run_dry_benchmark(models: list[str]) -> dict[str, list[dict]]:
    all_results = {}

    for model_id in models:
        print(f"\n  [DRY RUN] Model: {model_id}")
        results = []
        for sample in SAMPLES:
            asr = simulate_transcription(
                model_id, sample["sentence"], sample["locality"], sample["condition"]
            )
            metrics = compute_all_metrics(
                sample["sentence"], asr["transcript"], sample["locality"]
            )
            result = {
                "model": model_id,
                "index": sample["index"],
                "locality": sample["locality"],
                "condition": sample["condition"],
                "reference": sample["sentence"],
                "hypothesis": asr["transcript"],
                "filename": sample["filename"],
                "confidence": asr["confidence"],
                "latency_ms": asr["latency_ms"],
                "error": asr["error"],
                **metrics,
            }
            mark = "SUCCESS" if metrics["entity_exact"] else ("FUZZY" if metrics["entity_fuzzy"] else "FAILED")
            print(f"    {mark} {sample['locality']:<20} WER={metrics['wer']:.2f} "
                  f"EntitySim={metrics['entity_similarity']:.2f}")
            results.append(result)
        all_results[model_id] = results

    return all_results


def print_summary(all_results: dict):
    print(f"\n{'='*65}")
    print("  DRY RUN SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Model':<25} {'WER':>6} {'CER':>6} {'Exact%':>8} {'Fuzzy%':>8} {'Lat(ms)':>9}")
    print(f"  {'-'*65}")
    for model_id, results in all_results.items():
        agg = aggregate_results(results)
        valid = [r for r in results if not r.get("error")]
        avg_lat = sum(r["latency_ms"] for r in valid) / len(valid) if valid else 0
        print(f"  {model_id:<25} "
              f"{agg.get('mean_wer',0):>6.3f} "
              f"{agg.get('mean_cer',0):>6.3f} "
              f"{agg.get('accuracy_entity_exact',0)*100:>7.1f}% "
              f"{agg.get('accuracy_entity_fuzzy',0)*100:>7.1f}% "
              f"{avg_lat:>8.0f}ms")
    print()


def save_dry_results(all_results: dict):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"raw_results_{timestamp}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"  Dry-run results saved: {path}")

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    gt_path = AUDIO_DIR / "ground_truth.csv"
    if not gt_path.exists():
        with open(gt_path, "w", newline="", encoding="utf-8") as f:
            fields = ["index", "locality", "sentence", "condition", "filename",
                      "filepath", "status", "language"]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for s in SAMPLES:
                writer.writerow({
                    "index": s["index"],
                    "locality": s["locality"],
                    "sentence": s["sentence"],
                    "condition": s["condition"],
                    "filename": s["filename"],
                    "filepath": str(AUDIO_DIR / s["filename"]),
                    "status": "dry_run",
                    "language": "Hinglish",
                })
        print(f"  Ground truth template saved: {gt_path}")

    return path


if __name__ == "__main__":
    models = ["deepgram", "groq_whisper", "assemblyai", "sarvam"]
    print(f"\n{'='*65}")
    print("  ASR Shootout — DRY RUN (no API calls, simulated outputs)")
    print(f"{'='*65}")

    all_results = run_dry_benchmark(models)
    print_summary(all_results)
    save_dry_results(all_results)

    print("\n  Now generating charts...")
    from src.visualize import (
        plot_metric_comparison, plot_entity_heatmap,
        plot_latency, plot_condition_accuracy, plot_wer_vs_latency,
    )
    chart_dir = RESULTS_DIR / "charts"
    chart_dir.mkdir(exist_ok=True)
    plot_metric_comparison(all_results, chart_dir)
    plot_entity_heatmap(all_results, chart_dir)
    plot_latency(all_results, chart_dir)
    plot_condition_accuracy(all_results, chart_dir)
    plot_wer_vs_latency(all_results, chart_dir)
    print(f"\n  Done! Charts in: {chart_dir.resolve()}")
