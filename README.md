# ASR Shootout — Indian Conversational Speech Benchmark

A reproducible benchmarking pipeline for Automatic Speech Recognition (ASR) systems,
optimized for Hindi/Hinglish/Kannada conversational speech in noisy real-world conditions.
Built for evaluating ASR on a blue-collar hiring platform where candidates say things like
*"Haan, main Koramangala mein rehta hoon"*.

---

## Models Benchmarked

| Model | Provider | Type | Rationale |
|---|---|---|---|
| **Nova-2** | Deepgram | API (baseline) | Industry standard, low latency, Hindi telephony support |
| **Whisper-Large-v3** | Groq | OSS via API | Best open-source multilingual model, Groq LPU for speed |
| **Best** | AssemblyAI | API | Entity recognition focus, async batch |
| **Saarika v1** | Sarvam AI | API | Built specifically for Indian languages |

---

## Project Structure

```
asr_shootout/
├── .env.example              # Environment variable template
├── requirements.txt          # Python dependencies
├── REPORT.md                 # 3-page benchmark report
├── README.md                 # This file
│
├── audio_samples/            # Generated audio + ground truth
│   ├── 01_koramangala_quiet_room.wav
│   ├── ...
│   ├── ground_truth.csv      # Reference sentences per sample
│   └── metadata.json         # Full sample metadata
│
├── src/
│   ├── __init__.py
│   ├── benchmark.py          # Main pipeline orchestrator
│   ├── metrics.py            # WER, CER, EAR, NED implementations
│   ├── visualize.py          # Chart generation
│   └── models/
│       ├── __init__.py
│       ├── deepgram_asr.py   # Deepgram Nova-2 client
│       ├── groq_whisper_asr.py  # Groq Whisper client
│       ├── assemblyai_asr.py # AssemblyAI client
│       └── sarvam_asr.py     # Sarvam AI client
│
├── scripts/
│   ├── generate_audio.py     # TTS audio sample generator
│   └── dry_run.py            # Mock benchmark (no API keys needed)
│
├── results/                  # Auto-created; benchmark outputs
│   ├── raw_results_*.json
│   ├── per_sample_*.csv
│   ├── summary_*.csv
│   └── charts/
│       ├── 01_metric_comparison.png
│       ├── 02_entity_heatmap.png
│       ├── 03_latency_boxplot.png
│       ├── 04_condition_accuracy.png
│       └── 05_wer_vs_latency.png
│
└── notebooks/
    └── analysis.ipynb        # Exploratory analysis notebook
```

---

## Quick Start

### 1. Setup

```bash
cd asr_shootout
pip install -r requirements.txt

# Copy and fill in your API keys
cp .env.example .env
```

### 2. Generate Audio Samples

```bash
python scripts/generate_audio.py
```

Creates 20 WAV files in `./audio_samples/` with Hinglish sentences wrapping each
Bangalore locality name, with varied noise conditions.

### 3. Run the Benchmark

```bash
# Run all models from .env
python src/benchmark.py

# Run specific models
python src/benchmark.py --models deepgram groq_whisper

# Custom audio directory
python src/benchmark.py --audio_dir ./my_recordings
```

### 4. Generate Charts

```bash
python src/visualize.py
```

### 5. Dry Run

```bash
python scripts/dry_run.py
```

Runs the full pipeline with simulated (mocked) ASR outputs. Good for:
- Testing the pipeline structure
- Verifying metrics work correctly
- Generating example charts

---

## Environment Variables


```env
DEEPGRAM_API_KEY=...      # deepgram.com — free tier available
GROQ_API_KEY=...          # console.groq.com — free tier available
ASSEMBLYAI_API_KEY=...    # assemblyai.com — free tier available
SARVAM_API_KEY=...        # dashboard.sarvam.ai — free tier available

MODELS_TO_RUN=deepgram,groq_whisper,assemblyai
DEEPGRAM_MODEL=nova-2
GROQ_WHISPER_MODEL=whisper-large-v3
```

---

## Metrics Explained

| Metric | Description | Formula |
|---|---|---|
| **WER** | Word Error Rate | (S+D+I) / N_ref |
| **CER** | Character Error Rate | char-level edit distance / N_chars |
| **Entity Exact** | Did the locality appear verbatim? | Binary |
| **Entity Fuzzy** | Is similarity ≥ 0.80? | Levenshtein window match |
| **Entity Similarity** | Soft match score | 1 - (edit_distance / locality_len) |
| **Latency** | Wall-clock inference time | ms |

**Why EAR > WER for this task**: Entity Accuracy Rate directly measures what matters —
did the system correctly capture where the candidate lives? WER can look good even
when the locality is transcribed wrong.

---

## Adding a New Model

1. Create `src/models/my_model_asr.py` with a class that has `transcribe_file()` and `batch_transcribe()` methods
2. Register it in `src/benchmark.py` `get_model()` factory
3. Add API key to `.env`

---

## Recording Guidelines

If replacing synthetic audio with real recordings:
- Natural conversational Hindi/Hinglish sentences, not read off a script
- Vary conditions: quiet room, street/traffic noise, phone call, whispered, rushed
- Use phone mic — not a studio setup
- Name files: `{index:02d}_{locality_slug}_{condition}.wav`
- Update `audio_samples/ground_truth.csv` accordingly

---

## Output Files

After running `benchmark.py`:

- `results/raw_results_TIMESTAMP.json` — Full per-sample output per model
- `results/per_sample_TIMESTAMP.csv` — Flat CSV with all metrics
- `results/summary_TIMESTAMP.csv` — Aggregated metrics per model
- `results/charts/` — PNG visualizations

---

## Dependencies

```
deepgram-sdk    # Deepgram API
groq            # Groq (Whisper) API
assemblyai      # AssemblyAI API
requests        # Sarvam + HTTP calls
gtts            # TTS audio generation
pydub           # Audio format conversion + noise augmentation
jiwer           # WER reference implementation (cross-check)
editdistance    # Levenshtein distance
matplotlib      # Charts
seaborn         # Heatmaps
tabulate        # Console tables
python-dotenv   # .env loading
pandas, numpy   # Data manipulation
```
