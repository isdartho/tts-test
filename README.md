# Offline Python Text-to-Speech (TTS) Library with Kokoro-82M

A lightweight, pure offline, cross-platform Text-to-Speech library written in Python. It features built-in support for **Kokoro-82M** (state-of-the-art 82M open neural TTS model via `kokoro-onnx`), as well as native system engines (macOS `say`, Windows SAPI5, Linux `espeak`, and `pyttsx3` fallback).

---

## Key Features

- **Kokoro-82M Neural Engine**: State-of-the-art open weights neural speech synthesis with 54 studio-quality voices.
- **Pure Offline**: Operates 100% offline without requiring API keys or cloud services.
- **Auto-Model Downloader**: Automatically downloads model weights (`kokoro-v1.0.onnx` & `voices-v1.0.bin`) on first run.
- **Cross-Platform**: Seamless fallback across macOS, Windows, Linux, and ONNX Runtime.
- **Sync & Async API**: Native `asyncio` methods (`speak_async`, `save_async`, `stream_async`).
- **Audio Streaming**: Stream audio in byte chunks using sync or async generators.
- **Audio File Export**: Export 24kHz audio directly to `.wav`, `.aiff`, or `.mp3`.
- **Single-File Module**: Standalone `tts.py` file easy to drop into any project.

---

## Installation

```bash
pip install kokoro-onnx soundfile numpy
```

*(If `kokoro-onnx` is not installed, the library automatically falls back to native system engines like macOS `say` or Windows SAPI).*

---

## Quick Start

### 1. Using Kokoro-82M Neural Engine

```python
from tts import TTS

# Initialize Kokoro-82M TTS (downloads model on first use if needed)
tts = TTS(engine="kokoro", voice="af_heart", rate=180)

# Speak text aloud with high quality neural voice
tts.speak("Hello world! Kokoro-82M is synthesizing neural speech locally.")

# Save 24kHz audio file
tts.save("Saving neural speech to file.", "kokoro_output.wav")

# Switch voice (54 voices available!)
tts.set_voice("am_adam")
tts.speak("Switched to Adam's voice.")
```

### 2. Available Kokoro Neural Voices

Kokoro includes 54 voices across multiple languages:

| Language | Code | Sample Voices |
|---|---|---|
| **US English (Female)** | `en-us` | `af_heart`, `af_bella`, `af_nicole`, `af_sarah`, `af_nova`, `af_sky` |
| **US English (Male)** | `en-us` | `am_adam`, `am_michael`, `am_eric`, `am_echo`, `am_liam` |
| **UK English (Female)** | `en-gb` | `bf_emma`, `bf_alice`, `bf_isabella`, `bf_lily` |
| **UK English (Male)** | `en-gb` | `bm_george`, `bm_daniel`, `bm_lewis`, `bm_fable` |
| **Japanese** | `ja` | `jf_alpha`, `jf_nezumi`, `jm_kumo` |
| **Mandarin Chinese** | `zh` | `zf_xiaobei`, `zf_xiaoxiao`, `zm_yunjian` |
| **Hindi** | `hi` | `hf_alpha`, `hf_beta`, `hm_omega` |
| **Spanish / French / Italian** | `es`/`fr`/`it` | `ef_dora`, `ff_siwis`, `if_sara` |

To list all voices in code:
```python
for voice in tts.list_voices():
    print(voice.id, voice.language, voice.gender)
```

---

## Asynchronous Usage (`asyncio`)

```python
import asyncio
from tts import TTS

async def main():
    tts = TTS(engine="kokoro", voice="af_heart")

    # Non-blocking async speak
    await tts.speak_async("Hello from async Kokoro-82M!")

    # Non-blocking async file save
    await tts.save_async("Async speech export", "async_kokoro.wav")

    # Async audio streaming
    async for chunk in tts.stream_async("Streaming audio chunks asynchronously..."):
        pass

asyncio.run(main())
```

---

## Audio Streaming

```python
from tts import TTS

tts = TTS(engine="kokoro", voice="af_bella")

# Stream audio byte chunks (for real-time streaming, websockets, or speakers)
for chunk in tts.stream("Streaming text to chunked audio bytes...", chunk_size=4096):
    send_to_audio_player(chunk)
```

---

## CLI Usage

Run `tts.py` directly from the command line:

```bash
# Speak text with Kokoro-82M
python3 tts.py "Hello from command line Kokoro" -v af_heart

# Save speech to file
python3 tts.py "Save this neural audio" -v am_adam -o output.wav

# List available Kokoro voices
python3 tts.py --list-voices -e kokoro

# Use native system engine instead
python3 tts.py "System fallback" -e system
```

---

## Running Tests

Run the test suite:

```bash
python3 test_tts.py
```
