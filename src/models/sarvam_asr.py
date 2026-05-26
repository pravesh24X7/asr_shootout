import os
import time
from pathlib import Path
from typing import Optional
import requests
from dotenv import load_dotenv

load_dotenv()


class SarvamASR:
    BASE_URL = "https://api.sarvam.ai/speech-to-text"

    def __init__(
        self,
        api_key: Optional[str] = None,
        language_code: str = "hi-IN",
        model: str = "saarika:v1",
        with_timestamps: bool = False,
    ):
        self.api_key = api_key or os.getenv("SARVAM_API_KEY", "")
        if not self.api_key:
            raise ValueError("SARVAM_API_KEY not set. Check your .env file.")

        self.language_code = language_code
        self.model = model
        self.with_timestamps = with_timestamps
        self.model_name = f"sarvam_{model.replace(':', '_')}"

    def transcribe_file(self, audio_path: str | Path) -> dict:
        audio_path = Path(audio_path)
        if not audio_path.exists():
            return self._error_result(f"File not found: {audio_path}")

        headers = {
            "api-subscription-key": self.api_key,
        }

        try:
            t0 = time.time()

            with open(audio_path, "rb") as f:
                files = {"file": (audio_path.name, f, self._mime_type(audio_path))}
                data = {
                    "language_code": self.language_code,
                    "model": self.model,
                    "with_timestamps": str(self.with_timestamps).lower(),
                }
                response = requests.post(
                    self.BASE_URL,
                    headers=headers,
                    files=files,
                    data=data,
                    timeout=60,
                )

            latency_ms = (time.time() - t0) * 1000

            if response.status_code != 200:
                return self._error_result(
                    f"HTTP {response.status_code}: {response.text}",
                    latency_ms=latency_ms,
                )

            raw = response.json()
            transcript = raw.get("transcript", "") or ""

            return {
                "transcript": transcript,
                "confidence": 0.0, 
                "latency_ms": round(latency_ms, 2),
                "model": self.model_name,
                "raw": raw,
                "error": None,
            }

        except Exception as e:
            return self._error_result(str(e))

    @staticmethod
    def _mime_type(path: Path) -> str:
        ext = path.suffix.lower()
        mapping = {
            ".wav": "audio/wav",
            ".mp3": "audio/mpeg",
            ".m4a": "audio/mp4",
            ".ogg": "audio/ogg",
        }
        return mapping.get(ext, "audio/wav")

    def _error_result(self, msg: str, latency_ms: float = 0.0) -> dict:
        return {
            "transcript": "",
            "confidence": 0.0,
            "latency_ms": latency_ms,
            "model": self.model_name,
            "raw": {},
            "error": msg,
        }

    def batch_transcribe(self, audio_paths: list, verbose: bool = True) -> list[dict]:
        results = []
        for i, path in enumerate(audio_paths):
            if verbose:
                print(f"  [Sarvam {i+1}/{len(audio_paths)}] {Path(path).name}")
            result = self.transcribe_file(path)
            result["filename"] = Path(path).name
            if verbose:
                status = "✓" if not result["error"] else "✗"
                print(f"    {status} {result['transcript'][:60]}")
            results.append(result)
        return results
