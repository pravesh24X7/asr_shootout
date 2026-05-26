import os
import time
import json
from pathlib import Path
from typing import Optional
import requests
from dotenv import load_dotenv

load_dotenv()


class AssemblyAIASR:

    UPLOAD_URL = "https://api.assemblyai.com/v2/upload"
    TRANSCRIPT_URL = "https://api.assemblyai.com/v2/transcript"

    def __init__(
        self,
        api_key: Optional[str] = None,
        language_code: str = "hi",
        speech_model: str = "best",
        punctuate: bool = True,
    ):
        self.api_key = api_key or os.getenv("ASSEMBLYAI_API_KEY", "")
        if not self.api_key:
            raise ValueError("ASSEMBLYAI_API_KEY not set. Check your .env file.")

        self.language_code = language_code
        self.speech_model = speech_model
        self.punctuate = punctuate
        self.model_name = "assemblyai_best"

        self.headers = {
            "authorization": self.api_key,
            "content-type": "application/json",
        }

    def _upload_file(self, audio_path: Path) -> Optional[str]:
        with open(audio_path, "rb") as f:
            response = requests.post(
                self.UPLOAD_URL,
                headers={"authorization": self.api_key},
                data=f,
                timeout=120,
            )
        if response.status_code == 200:
            return response.json()["upload_url"]
        raise RuntimeError(f"Upload failed: {response.status_code} {response.text}")

    def _submit_transcript(self, audio_url: str) -> str:
        payload = {
            "audio_url": audio_url,
            "speech_model": self.speech_model,
            "punctuate": self.punctuate,
        }
        if self.language_code:
            payload["language_code"] = self.language_code

        response = requests.post(
            self.TRANSCRIPT_URL,
            json=payload,
            headers=self.headers,
            timeout=30,
        )
        if response.status_code == 200:
            return response.json()["id"]
        raise RuntimeError(f"Submission failed: {response.status_code} {response.text}")

    def _poll_transcript(self, job_id: str, timeout: int = 120) -> dict:
        url = f"{self.TRANSCRIPT_URL}/{job_id}"
        deadline = time.time() + timeout

        while time.time() < deadline:
            response = requests.get(url, headers=self.headers, timeout=30)
            data = response.json()
            status = data.get("status")

            if status == "completed":
                return data
            elif status == "error":
                raise RuntimeError(f"AssemblyAI error: {data.get('error')}")
            time.sleep(2)

        raise TimeoutError(f"Transcript {job_id} did not complete within {timeout}s")

    def transcribe_file(self, audio_path: str | Path) -> dict:

        audio_path = Path(audio_path)
        if not audio_path.exists():
            return self._error_result(f"File not found: {audio_path}")

        try:
            t0 = time.time()

            upload_url = self._upload_file(audio_path)

            job_id = self._submit_transcript(upload_url)

            raw = self._poll_transcript(job_id)

            latency_ms = (time.time() - t0) * 1000

            transcript = raw.get("text", "") or ""
            confidence = raw.get("confidence", 0.0) or 0.0

            return {
                "transcript": transcript,
                "confidence": round(float(confidence), 4),
                "latency_ms": round(latency_ms, 2),
                "model": self.model_name,
                "raw": raw,
                "error": None,
            }

        except Exception as e:
            return self._error_result(str(e))

    @staticmethod
    def _error_result(msg: str, latency_ms: float = 0.0) -> dict:
        return {
            "transcript": "",
            "confidence": 0.0,
            "latency_ms": latency_ms,
            "model": "assemblyai",
            "raw": {},
            "error": msg,
        }

    def batch_transcribe(self, audio_paths: list, verbose: bool = True) -> list[dict]:
        results = []
        for i, path in enumerate(audio_paths):
            if verbose:
                print(f"  [AssemblyAI {i+1}/{len(audio_paths)}] {Path(path).name}")
            result = self.transcribe_file(path)
            result["filename"] = Path(path).name
            if verbose:
                status = "SUCCESS" if not result["error"] else "FAILED"
                print(f"    {status} {result['transcript'][:60]}")
            results.append(result)
        return results
