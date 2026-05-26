import os
import sys
import json
import glob
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import seaborn as sns
    HAS_SNS = True
except ImportError:
    HAS_SNS = False

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.metrics import aggregate_results

RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "./results"))

MODEL_COLORS = {
    "deepgram": "#4C72B0",
    "groq_whisper": "#DD8452",
    "assemblyai": "#55A868",
    "sarvam": "#C44E52",
    "default": "#8172B3",
}


def color_for(model: str) -> str:
    for key, col in MODEL_COLORS.items():
        if key in model.lower():
            return col
    return MODEL_COLORS["default"]



def load_latest_results(results_dir: Path = RESULTS_DIR) -> dict:
    pattern = str(results_dir / "raw_results_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No raw_results_*.json found in {results_dir}. "
            "Run the benchmark first: python src/benchmark.py"
        )
    latest = files[-1]
    print(f"  Loading: {latest}")
    with open(latest, "r", encoding="utf-8") as f:
        return json.load(f)



def plot_metric_comparison(all_results: dict, output_dir: Path):
    if not HAS_MPL:
        print("  [SKIP] matplotlib not installed")
        return

    models = list(all_results.keys())
    metrics = {
        "WER (↓)": "mean_wer",
        "CER (↓)": "mean_cer",
        "Entity Exact (↑)": "accuracy_entity_exact",
        "Entity Fuzzy (↑)": "accuracy_entity_fuzzy",
    }

    agg_data = {}
    for model_id, results in all_results.items():
        if results:
            agg_data[model_id] = aggregate_results(results)

    n_metrics = len(metrics)
    n_models = len(models)
    x = np.arange(n_metrics)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    for i, model_id in enumerate(models):
        agg = agg_data.get(model_id, {})
        vals = [agg.get(key, 0) or 0 for key in metrics.values()]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width * 0.9,
                      label=model_id, color=color_for(model_id),
                      edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(list(metrics.keys()), fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("ASR Benchmark — Model Comparison", fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="upper right", framealpha=0.8)
    ax.grid(axis="y", alpha=0.4, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    path = output_dir / "01_metric_comparison.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")



def plot_entity_heatmap(all_results: dict, output_dir: Path):
    if not HAS_MPL:
        return

    models = list(all_results.keys())
    localities = []
    seen = set()
    for results in all_results.values():
        for r in results:
            loc = r["locality"]
            if loc not in seen:
                localities.append(loc)
                seen.add(loc)

    matrix = np.zeros((len(models), len(localities)))
    for i, model_id in enumerate(models):
        results = all_results.get(model_id, [])
        loc_map = {r["locality"]: r["entity_similarity"] for r in results}
        for j, loc in enumerate(localities):
            matrix[i][j] = loc_map.get(loc, 0.0)

    fig, ax = plt.subplots(figsize=(max(14, len(localities) * 0.7), max(4, len(models) * 1.2)))
    fig.patch.set_facecolor("#F8F9FA")

    cmap = "RdYlGn"
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(localities)))
    ax.set_xticklabels(localities, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=10)

    for i in range(len(models)):
        for j in range(len(localities)):
            val = matrix[i][j]
            color = "white" if val < 0.4 or val > 0.8 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=color)

    plt.colorbar(im, ax=ax, shrink=0.8, label="Entity Similarity")
    ax.set_title("Entity Similarity by Model × Locality", fontsize=13, fontweight="bold", pad=12)

    path = output_dir / "02_entity_heatmap.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")



def plot_latency(all_results: dict, output_dir: Path):
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    model_latencies = {}
    for model_id, results in all_results.items():
        lats = [r["latency_ms"] for r in results if not r.get("error") and r.get("latency_ms", 0) > 0]
        if lats:
            model_latencies[model_id] = lats

    if not model_latencies:
        print("  [SKIP] No latency data available")
        return

    bp = ax.boxplot(
        list(model_latencies.values()),
        labels=list(model_latencies.keys()),
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 2},
    )

    for patch, model_id in zip(bp["boxes"], model_latencies.keys()):
        patch.set_facecolor(color_for(model_id))
        patch.set_alpha(0.7)

    # Overlay scatter
    for i, (model_id, lats) in enumerate(model_latencies.items(), 1):
        jitter = np.random.normal(0, 0.05, len(lats))
        ax.scatter([i + j for j in jitter], lats,
                   alpha=0.5, s=30, color=color_for(model_id), zorder=5)

    ax.set_ylabel("Latency (ms)", fontsize=11)
    ax.set_title("ASR Latency Distribution by Model", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.4, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    path = output_dir / "03_latency_boxplot.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")



def plot_condition_accuracy(all_results: dict, output_dir: Path):
    if not HAS_MPL:
        return

    conditions = set()
    for results in all_results.values():
        for r in results:
            conditions.add(r["condition"])
    conditions = sorted(conditions)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    x = np.arange(len(conditions))
    n = len(all_results)
    width = 0.8 / n

    for i, (model_id, results) in enumerate(all_results.items()):
        accuracies = []
        for cond in conditions:
            cond_r = [r for r in results if r["condition"] == cond]
            if cond_r:
                acc = sum(r["entity_exact"] for r in cond_r) / len(cond_r)
            else:
                acc = 0.0
            accuracies.append(acc)

        offset = (i - n / 2 + 0.5) * width
        ax.bar(x + offset, accuracies, width * 0.9,
               label=model_id, color=color_for(model_id),
               edgecolor="white", linewidth=0.5, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Entity Exact Accuracy", fontsize=11)
    ax.set_title("Entity Accuracy by Recording Condition", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.8)
    ax.grid(axis="y", alpha=0.4, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    path = output_dir / "04_condition_accuracy.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")



def plot_wer_vs_latency(all_results: dict, output_dir: Path):
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    for model_id, results in all_results.items():
        valid = [r for r in results if not r.get("error")]
        if not valid:
            continue
        agg = aggregate_results(valid)
        avg_wer = agg.get("mean_wer", 0) or 0
        avg_lat = sum(r["latency_ms"] for r in valid) / len(valid)

        ax.scatter(avg_lat, avg_wer, s=200, color=color_for(model_id),
                   zorder=5, label=model_id, edgecolors="white", linewidths=1.5)
        ax.annotate(model_id, (avg_lat, avg_wer),
                    textcoords="offset points", xytext=(8, 4), fontsize=9)

    ax.set_xlabel("Avg Latency (ms) →  faster is better", fontsize=11)
    ax.set_ylabel("Avg WER →  lower is better", fontsize=11)
    ax.set_title("Accuracy vs Speed Tradeoff", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.4, linestyle="--")
    ax.legend(framealpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.text(0.05, 0.95, "← Better (low WER, low latency)",
            transform=ax.transAxes, fontsize=8, color="green", va="top")

    path = output_dir / "05_wer_vs_latency.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")



def main():
    parser = argparse.ArgumentParser(description="Generate ASR benchmark visualizations")
    parser.add_argument("--results", type=str, default=None,
                        help="Path to raw_results_*.json (default: latest in results/)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for charts (default: results/charts/)")
    args = parser.parse_args()

    if not HAS_MPL:
        print("[ERROR] matplotlib not installed. Run: pip install matplotlib seaborn")
        sys.exit(1)

    # Load results
    if args.results:
        with open(args.results, "r") as f:
            all_results = json.load(f)
    else:
        all_results = load_latest_results()

    output_dir = Path(args.output_dir or RESULTS_DIR / "charts")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Generating charts → {output_dir.resolve()}\n")

    plot_metric_comparison(all_results, output_dir)
    plot_entity_heatmap(all_results, output_dir)
    plot_latency(all_results, output_dir)
    plot_condition_accuracy(all_results, output_dir)
    plot_wer_vs_latency(all_results, output_dir)

    print(f"\n  All charts saved to {output_dir}")


if __name__ == "__main__":
    main()
