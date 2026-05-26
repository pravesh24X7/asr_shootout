
import os
import time
import json
from pathlib import Path
from typing import Optional
import requests
from dotenv import load_dotenv

load_dotenv()


class DeepgramASR:

    BASE_URL = "https://api.deepgram.com/v1/listen"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "nova-2",
        language: str = "hi",
        punctuate: bool = True,
        diarize: bool = False,
    ):
        self.api_key = api_key or os.getenv("DEEPGRAM_API_KEY", "")
        if not self.api_key:
            raise ValueError("DEEPGRAM_API_KEY not set. Check your .env file.")

        self.model = os.getenv("DEEPGRAM_MODEL", model)
        self.language = language
        self.punctuate = punctuate
        self.diarize = diarize
        self.model_name = f"deepgram_{self.model}"

    def transcribe_file(self, audio_path: str | Path) -> dict:

        audio_path = Path(audio_path)
        if not audio_path.exists():
            return self._error_result(f"File not found: {audio_path}")

        params = {
            "model": self.model,
            "language": self.language,
            "punctuate": str(self.punctuate).lower(),
            "diarize": str(self.diarize).lower(),
            "smart_format": "true",
        }

        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": self._content_type(audio_path),
        }

        try:
            t0 = time.time()
            with open(audio_path, "rb") as f:
                response = requests.post(
                    self.BASE_URL,
                    headers=headers,
                    params=params,
                    data=f,
                    timeout=60,
                )
            latency_ms = (time.time() - t0) * 1000

            if response.status_code != 200:
                return self._error_result(
                    f"HTTP {response.status_code}: {response.text}",
                    latency_ms=latency_ms,
                )

            raw = response.json()
            transcript, confidence = self._parse_response(raw)

            return {
                "transcript": transcript,
                "confidence": confidence,
                "latency_ms": round(latency_ms, 2),
                "model": self.model_name,
                "raw": raw,
                "error": None,
            }

        except requests.exceptions.Timeout:
            return self._error_result("Request timed out after 60s")
        except Exception as e:
            return self._error_result(str(e))

    def _parse_response(self, raw: dict) -> tuple[str, float]:
        try:
            results = raw["results"]["channels"][0]["alternatives"][0]
            transcript = results.get("transcript", "")
            confidence = results.get("confidence", 0.0)
            return transcript, confidence
        except (KeyError, IndexError):
            return "", 0.0

    @staticmethod
    def _content_type(path: Path) -> str:
        ext = path.suffix.lower()
        mapping = {
            ".wav": "audio/wav",
            ".mp3": "audio/mpeg",
            ".m4a": "audio/mp4",
            ".ogg": "audio/ogg",
            ".flac": "audio/flac",
            ".webm": "audio/webm",
        }
        return mapping.get(ext, "audio/wav")

    @staticmethod
    def _error_result(msg: str, latency_ms: float = 0.0) -> dict:
        return {
            "transcript": "",
            "confidence": 0.0,
            "latency_ms": latency_ms,
            "model": "deepgram",
            "raw": {},
            "error": msg,
        }

    def batch_transcribe(self, audio_paths: list, verbose: bool = True) -> list[dict]:
        results = []
        for i, path in enumerate(audio_paths):
            if verbose:
                print(f"  [Deepgram {i+1}/{len(audio_paths)}] {Path(path).name}")
            result = self.transcribe_file(path)
            result["filename"] = Path(path).name
            if verbose:
                status = "✓" if not result["error"] else "✗"
                print(f"    {status} {result['transcript'][:60]}...")
            results.append(result)
        return results
