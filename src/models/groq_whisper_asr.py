import os
import time
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class GroqWhisperASR:

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "whisper-large-v3",
        language: Optional[str] = "hi",
        temperature: float = 0.0,
        response_format: str = "verbose_json",
    ):
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not set. Check your .env file.")

        self.model = os.getenv("GROQ_WHISPER_MODEL", model)
        self.language = language
        self.temperature = temperature
        self.response_format = response_format
        self.model_name = f"groq_{self.model.replace('-', '_')}"
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                from groq import Groq
                self._client = Groq(api_key=self.api_key)
            except ImportError:
                raise ImportError("Install groq SDK: pip install groq")
        return self._client

    def transcribe_file(self, audio_path: str | Path) -> dict:

        audio_path = Path(audio_path)
        if not audio_path.exists():
            return self._error_result(f"File not found: {audio_path}")

        try:
            t0 = time.time()

            with open(audio_path, "rb") as f:
                kwargs = dict(
                    file=(audio_path.name, f, self._mime_type(audio_path)),
                    model=self.model,
                    temperature=self.temperature,
                    response_format=self.response_format,
                )
                if self.language:
                    kwargs["language"] = self.language

                transcription = self.client.audio.transcriptions.create(**kwargs)

            latency_ms = (time.time() - t0) * 1000

            raw = transcription.model_dump() if hasattr(transcription, "model_dump") else {}
            transcript = getattr(transcription, "text", "") or raw.get("text", "")

            segments = getattr(transcription, "segments", None) or raw.get("segments", [])
            if segments:
                avg_conf = sum(s.get("avg_logprob", 0) for s in segments) / len(segments)
                confidence = float(min(1.0, max(0.0, 2 ** avg_conf)))
            else:
                confidence = 0.0

            lang_detected = getattr(transcription, "language", None) or raw.get("language", "unknown")

            return {
                "transcript": transcript,
                "confidence": round(confidence, 4),
                "latency_ms": round(latency_ms, 2),
                "model": self.model_name,
                "language_detected": lang_detected,
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
            ".flac": "audio/flac",
            ".webm": "audio/webm",
        }
        return mapping.get(ext, "audio/wav")

    def _error_result(self, msg: str, latency_ms: float = 0.0) -> dict:
        return {
            "transcript": "",
            "confidence": 0.0,
            "latency_ms": latency_ms,
            "model": self.model_name,
            "language_detected": "unknown",
            "raw": {},
            "error": msg,
        }

    def batch_transcribe(self, audio_paths: list, verbose: bool = True) -> list[dict]:
        results = []
        for i, path in enumerate(audio_paths):
            if verbose:
                print(f"  [Groq Whisper {i+1}/{len(audio_paths)}] {Path(path).name}")
            result = self.transcribe_file(path)
            result["filename"] = Path(path).name
            if verbose:
                status = "✓" if not result["error"] else "✗"
                print(f"    {status} [{result.get('language_detected','?')}] {result['transcript'][:60]}")
            results.append(result)
        return results
