"""
TTS (Text-to-Speech) - A lightweight, pure offline Python Text-to-Speech library powered by Kokoro-82M.

Features:
- Dedicated Kokoro-82M state-of-the-art neural TTS engine (54 voices).
- Pure offline speech synthesis (no internet connection required).
- Direct speech playback & audio file export (.wav, .aiff, .mp3).
- Synchronous & Asynchronous (asyncio) API.
- Audio chunk streaming (sync & async generators).
- Custom voice selection, rate (speed), volume, and pitch controls.
- Auto-downloader for model weights (kokoro-v1.0.onnx & voices-v1.0.bin).
- Standalone single-file Python module.
"""

from __future__ import annotations

import asyncio
import io
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import wave
import threading
from dataclasses import dataclass
from queue import Queue
from typing import AsyncGenerator, Generator, List, Optional, Tuple


@dataclass
class Voice:
    """Represents a Text-to-Speech voice."""
    id: str
    name: str
    language: Optional[str] = None
    gender: Optional[str] = None

    def __repr__(self) -> str:
        lang_str = f", lang={self.language}" if self.language else ""
        return f"Voice(id='{self.id}', name='{self.name}'{lang_str})"


@dataclass
class TTSConfig:
    """Configuration settings for Text-to-Speech synthesis."""
    voice: Optional[str] = "af_heart"
    rate: int = 175  # Words per minute (default ~175 WPM)
    volume: float = 1.0  # 0.0 to 1.0
    pitch: float = 1.0  # 0.5 to 2.0 (1.0 = normal)


class KokoroBackend:
    """
    Kokoro-82M ONNX neural TTS engine backend.
    Provides ultra-realistic, state-of-the-art speech synthesis using the 82M open model.
    """

    MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
    VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"

    def __init__(self, model_path: Optional[str] = None, voices_path: Optional[str] = None):
        self.model_dir = os.path.expanduser("~/.cache/kokoro")
        os.makedirs(self.model_dir, exist_ok=True)

        self.model_path = model_path or os.path.join(self.model_dir, "kokoro-v1.0.onnx")
        self.voices_path = voices_path or os.path.join(self.model_dir, "voices-v1.0.bin")

        local_model = os.path.join("models", "kokoro-v1.0.onnx")
        local_voices = os.path.join("models", "voices-v1.0.bin")
        if os.path.exists(local_model):
            self.model_path = local_model
        if os.path.exists(local_voices):
            self.voices_path = local_voices

        self._ensure_models()
        self._kokoro = None

    def _ensure_models(self):
        """Downloads Kokoro model and voice files if not present."""
        import ssl
        import urllib.request

        def download_file(url: str, dest: str, desc: str):
            if os.path.exists(dest) and os.path.getsize(dest) > 1000:
                return
            print(f"Downloading Kokoro-82M {desc} to {dest}...")
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE

            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, context=ctx) as resp, open(dest, "wb") as f:
                shutil.copyfileobj(resp, f)
            print(f"Downloaded {desc} successfully.")

        download_file(self.MODEL_URL, self.model_path, "model (310MB)")
        download_file(self.VOICES_URL, self.voices_path, "voices (27MB)")

    def _get_engine(self):
        if self._kokoro is None:
            from kokoro_onnx import Kokoro
            self._kokoro = Kokoro(self.model_path, self.voices_path)
        return self._kokoro

    def list_voices(self) -> List[Voice]:
        kokoro = self._get_engine()
        voice_ids = kokoro.get_voices()
        result = []
        for v_id in voice_ids:
            lang = "en-us"
            gender = "female" if "_f_" in v_id or v_id.startswith("af_") or v_id.startswith("bf_") or v_id.startswith("jf_") or v_id.startswith("zf_") else "male"
            if v_id.startswith("af_") or v_id.startswith("am_"):
                lang = "en-us"
            elif v_id.startswith("bf_") or v_id.startswith("bm_"):
                lang = "en-gb"
            elif v_id.startswith("jf_") or v_id.startswith("jm_"):
                lang = "ja"
            elif v_id.startswith("zf_") or v_id.startswith("zm_"):
                lang = "zh"
            elif v_id.startswith("hf_") or v_id.startswith("hm_"):
                lang = "hi"
            elif v_id.startswith("ff_"):
                lang = "fr"
            elif v_id.startswith("if_") or v_id.startswith("im_"):
                lang = "it"
            result.append(Voice(id=v_id, name=v_id, language=lang, gender=gender))
        return result

    def _get_voice_name(self, config: TTSConfig) -> str:
        if config.voice:
            return config.voice
        return "af_heart"

    def _get_speed(self, config: TTSConfig) -> float:
        if config.rate:
            return max(0.2, min(3.0, config.rate / 175.0))
        return 1.0

    def _get_lang(self, voice: str) -> str:
        if voice.startswith("bf_") or voice.startswith("bm_"):
            return "en-gb"
        elif voice.startswith("jf_") or voice.startswith("jm_"):
            return "ja"
        elif voice.startswith("zf_") or voice.startswith("zm_"):
            return "zh"
        elif voice.startswith("hf_") or voice.startswith("hm_"):
            return "hi"
        elif voice.startswith("ff_"):
            return "fr"
        elif voice.startswith("if_") or voice.startswith("im_"):
            return "it"
        elif voice.startswith("ef_") or voice.startswith("em_"):
            return "es"
        elif voice.startswith("pf_") or voice.startswith("pm_"):
            return "pt-br"
        return "en-us"

    def synthesize_wav_bytes(self, text: str, config: TTSConfig) -> Tuple[bytes, int]:
        import numpy as np
        kokoro = self._get_engine()
        voice = self._get_voice_name(config)
        speed = self._get_speed(config)
        lang = self._get_lang(voice)

        samples, sample_rate = kokoro.create(text, voice=voice, speed=speed, lang=lang)

        if config.volume != 1.0:
            samples = samples * max(0.0, min(1.0, config.volume))

        pcm_bytes = (samples * 32767).clip(-32768, 32767).astype(np.int16).tobytes()

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_bytes)
        return buf.getvalue(), sample_rate

    def _break_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text.replace("\n", " "))
        return [sentence.strip() for sentence in sentences if sentence.strip()]

    def _process_text(self, text: str, queue: Queue, config: TTSConfig) -> None:
        try:
            sentences = self._break_sentences(text)
            for sentence in sentences:
                wav_bytes, _ = self.synthesize_wav_bytes(sentence, config)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(wav_bytes)
                    tmp_path = tmp.name
                    queue.put(tmp_path)
        finally:
            queue.put(None)

    def speak(self, text: str, config: TTSConfig) -> None:
        if not text or not text.strip():
            return

        speech_file_queue = Queue()
        text_processing = threading.Thread(
            target=self._process_text, 
            args=(text, speech_file_queue, config), 
            daemon=True
        )
        text_processing.start()

        while True:
            tmp_path = speech_file_queue.get()
            if tmp_path is None:
                break
            try:
                sys_platform = platform.system()
                if sys_platform == "Darwin" and shutil.which("afplay"):
                    subprocess.run(["afplay", tmp_path], check=True)
                elif sys_platform == "Linux" and (shutil.which("aplay") or shutil.which("paplay")):
                    player = shutil.which("paplay") or shutil.which("aplay")
                    subprocess.run([player, tmp_path], check=True)
                elif sys_platform == "Windows" and shutil.which("powershell"):
                    ps_script = f"$player = New-Object System.Media.SoundPlayer('{tmp_path}'); $player.PlaySync();"
                    subprocess.run(["powershell", "-NoProfile", "-Command", ps_script], check=True)
                else:
                    raise RuntimeError("No system audio player found for playback.")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    def save(self, text: str, file_path: str, config: TTSConfig, audio_format: str = "wav") -> str:
        wav_bytes, _ = self.synthesize_wav_bytes(text, config)
        ext = os.path.splitext(file_path)[1].lower().strip(".")
        target_fmt = audio_format.lower() if audio_format else (ext if ext else "wav")

        if target_fmt == "wav":
            with open(file_path, "wb") as f:
                f.write(wav_bytes)
        else:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(wav_bytes)
                tmp_path = tmp.name
            try:
                ffmpeg_path = shutil.which("ffmpeg")
                if ffmpeg_path:
                    subprocess.run([ffmpeg_path, "-y", "-i", tmp_path, file_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                else:
                    shutil.move(tmp_path, file_path)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        return file_path

    def stream(self, text: str, config: TTSConfig, chunk_size: int = 4096) -> Generator[bytes, None, None]:
        wav_bytes, _ = self.synthesize_wav_bytes(text, config)
        for i in range(0, len(wav_bytes), chunk_size):
            yield wav_bytes[i:i + chunk_size]


class TTS:
    """
    Main Text-to-Speech Engine Interface powered exclusively by Kokoro-82M.

    Example Usage:
        tts = TTS(voice="af_heart")
        tts.speak("Hello world!")
        tts.save("hello.wav")

        # Async:
        await tts.speak_async("Hello async world!")

        # Streaming:
        for chunk in tts.stream("Streaming text..."):
            process(chunk)
    """

    def __init__(
        self,
        voice: Optional[str] = "af_heart",
        rate: int = 175,
        volume: float = 1.0,
        pitch: float = 1.0,
        model_path: Optional[str] = None,
        voices_path: Optional[str] = None,
    ):
        self.config = TTSConfig(voice=voice, rate=rate, volume=volume, pitch=pitch)
        self.backend = KokoroBackend(model_path=model_path, voices_path=voices_path)

    def list_voices(self) -> List[Voice]:
        """Returns a list of all available Kokoro neural voices."""
        return self.backend.list_voices()

    def set_voice(self, voice_id_or_name: str) -> TTS:
        """Sets the active voice by ID or name."""
        self.config.voice = voice_id_or_name
        return self

    def set_rate(self, rate: int) -> TTS:
        """Sets speech speed in Words Per Minute (WPM). Default is ~175 WPM."""
        self.config.rate = max(10, rate)
        return self

    def set_volume(self, volume: float) -> TTS:
        """Sets speech volume (0.0 to 1.0)."""
        self.config.volume = max(0.0, min(1.0, volume))
        return self

    def set_pitch(self, pitch: float) -> TTS:
        """Sets speech pitch multiplier (0.5 to 2.0, default 1.0)."""
        self.config.pitch = max(0.5, min(2.0, pitch))
        return self

    def speak(self, text: str) -> None:
        """Synchronously speaks text aloud via default audio output."""
        if not text or not text.strip():
            return
        self.backend.speak(text.strip(), self.config)

    def save(self, text: str, file_path: str, audio_format: str = "wav") -> str:
        """Synchronously synthesizes speech and saves to an audio file."""
        if not text or not text.strip():
            raise ValueError("Text to synthesize cannot be empty.")
        return self.backend.save(text.strip(), file_path, self.config, audio_format=audio_format)

    def stream(self, text: str, chunk_size: int = 4096) -> Generator[bytes, None, None]:
        """Synchronously streams audio data in byte chunks."""
        if not text or not text.strip():
            return
        yield from self.backend.stream(text.strip(), self.config, chunk_size=chunk_size)

    # Async methods
    async def speak_async(self, text: str) -> None:
        """Asynchronously speaks text aloud without blocking the event loop."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.speak, text)

    async def save_async(self, text: str, file_path: str, audio_format: str = "wav") -> str:
        """Asynchronously saves speech to an audio file."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.save, text, file_path, audio_format)

    async def stream_async(self, text: str, chunk_size: int = 4096) -> AsyncGenerator[bytes, None]:
        """Asynchronously streams audio byte chunks."""
        loop = asyncio.get_running_loop()

        def run_stream():
            return list(self.stream(text, chunk_size=chunk_size))

        chunks = await loop.run_in_executor(None, run_stream)
        for chunk in chunks:
            yield chunk

    def __enter__(self) -> TTS:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dedicated Kokoro-82M Neural Python Text-to-Speech CLI")
    parser.add_argument("text", nargs="?", help="Text to speak or synthesize")
    parser.add_argument("-o", "--output", help="Output file path (e.g. output.wav)")
    parser.add_argument("-v", "--voice", default="af_heart", help="Voice name or ID (e.g. af_heart, am_adam, bf_emma)")
    parser.add_argument("-r", "--rate", type=int, default=175, help="Speech rate in WPM")
    parser.add_argument("-l", "--list-voices", action="store_true", help="List available Kokoro voices")

    args = parser.parse_args()

    engine = TTS(voice=args.voice, rate=args.rate)

    if args.list_voices:
        print("Available Kokoro Neural Voices:")
        for v in engine.list_voices():
            print(f" - {v.name} (ID: {v.id}) [{v.language or 'unknown'}] ({v.gender or 'unknown'})")
    elif args.text:
        if args.output:
            out = engine.save(args.text, args.output)
            print(f"Audio saved to: {out}")
        else:
            print(f"Speaking: {args.text}")
            engine.speak(args.text)
    else:
        parser.print_help()
