import os
import sys
import random
import json
import csv
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

AUDIO_DIR = Path(os.getenv("AUDIO_DIR", "./audio_samples"))
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

LOCALITIES = [
    "Koramangala",
    "Indiranagar",
    "Whitefield",
    "Electronic City",
    "Marathahalli",
    "Jayanagar",
    "Rajajinagar",
    "Hebbal",
    "Yelahanka",
    "Banashankari",
    "HSR Layout",
    "BTM Layout",
    "Majestic",
    "Silk Board",
    "Bellandur",
    "Sarjapur",
    "Bommanahalli",
    "KR Puram",
    "Peenya",
    "Yeshwanthpur",
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
    "Mujhe {loc} jaana hai",
    "Main {loc} se aata hoon",
]

CONDITIONS = [
    "quiet_room",
    "street_noise",
    "phone_call",
    "rushed",
    "whispered",
]


def generate_audio_gtts(text: str, output_path: Path, lang: str = "hi") -> bool:
    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang=lang, slow=False)
        mp3_path = output_path.with_suffix(".mp3")
        tts.save(str(mp3_path))

        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_mp3(str(mp3_path))
            audio.export(str(output_path), format="wav")
            mp3_path.unlink(missing_ok=True)
        except Exception:
            import shutil
            shutil.copy(str(mp3_path), str(output_path.with_suffix(".mp3")))
            mp3_path.unlink(missing_ok=True)
            return False

        return True
    except ImportError:
        print("  [!] gTTS not installed. Run: pip install gtts")
        return False
    except Exception as e:
        print(f"  [!] gTTS error: {e}")
        return False


def apply_noise_augmentation(wav_path: Path, condition: str) -> None:
    try:
        from pydub import AudioSegment
        from pydub.effects import normalize
        import numpy as np

        audio = AudioSegment.from_wav(str(wav_path))

        if condition == "phone_call":
            audio = audio.set_frame_rate(8000).set_frame_rate(16000)
            audio = audio - 3  

        elif condition == "street_noise":
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            noise = np.random.normal(0, 300, len(samples)).astype(np.float32)
            noisy = np.clip(samples + noise, -32768, 32767).astype(np.int16)
            noisy_audio = audio._spawn(noisy.tobytes())
            audio = noisy_audio

        elif condition == "whispered":
            audio = audio - 12

        elif condition == "rushed":
            audio = audio.speedup(playback_speed=1.3)

        audio = normalize(audio)
        audio.export(str(wav_path), format="wav")

    except Exception as e:
        print(f"  [!] Noise augmentation skipped ({condition}): {e}")


def generate_dataset() -> list[dict]:
    metadata = []
    random.seed(42)

    print(f"\n{'='*60}")
    print("  ASR Shootout — Audio Dataset Generator")
    print(f"{'='*60}")
    print(f"  Output directory: {AUDIO_DIR.resolve()}")
    print(f"  Samples to generate: {len(LOCALITIES)}")
    print()

    for i, locality in enumerate(LOCALITIES):
        template = random.choice(TEMPLATES)
        condition = CONDITIONS[i % len(CONDITIONS)]
        sentence = template.format(loc=locality)

        slug = locality.lower().replace(" ", "_")
        filename = f"{i+1:02d}_{slug}_{condition}.wav"
        wav_path = AUDIO_DIR / filename

        print(f"  [{i+1:02d}/20] {locality}")
        print(f"         Sentence  : \"{sentence}\"")
        print(f"         Condition : {condition}")
        print(f"         File      : {filename}")

        success = generate_audio_gtts(sentence, wav_path, lang="hi")

        if success and wav_path.exists():
            apply_noise_augmentation(wav_path, condition)
            status = "ok"
        else:
            mp3_path = wav_path.with_suffix(".mp3")
            if mp3_path.exists():
                filename = filename.replace(".wav", ".mp3")
                wav_path = mp3_path
                status = "ok_mp3"
            else:
                status = "failed"

        metadata.append({
            "index": i + 1,
            "locality": locality,
            "sentence": sentence,
            "condition": condition,
            "filename": filename,
            "filepath": str(AUDIO_DIR / filename),
            "status": status,
            "language": "Hinglish",
        })

        print(f"         Status    : {status}\n")

    gt_path = AUDIO_DIR / "ground_truth.csv"
    with open(gt_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=metadata[0].keys())
        writer.writeheader()
        writer.writerows(metadata)

    meta_path = AUDIO_DIR / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"{'='*60}")
    print(f"  Ground truth saved: {gt_path}")
    print(f"  Metadata saved    : {meta_path}")
    print(f"  Total generated   : {sum(1 for m in metadata if m['status'].startswith('ok'))}/20")
    print(f"{'='*60}\n")

    return metadata


if __name__ == "__main__":
    meta = generate_dataset()
    ok = sum(1 for m in meta if m["status"].startswith("ok"))
    if ok < 20:
        print(f"[WARNING] Only {ok}/20 samples generated successfully.")
        print("  Make sure gTTS and pydub are installed:")
        print("  pip install gtts pydub")
        sys.exit(1)
    else:
        print(f"[SUCCESS] All {ok} samples generated.")
