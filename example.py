"""
Example Usage of Kokoro-82M Neural Text-to-Speech Library (tts.py)
"""

import asyncio
from tts import TTS


def kokoro_demo():
    print("--- 1. Kokoro-82M Neural TTS Demo ---")
    # Initialize Kokoro neural TTS with high quality female voice 'af_heart'
    tts = TTS(engine="kokoro", voice="af_heart", rate=180)

    # List available Kokoro neural voices
    voices = tts.list_voices()
    print(f"Found {len(voices)} high-quality Kokoro neural voices.")
    print("Sample voices:", [v.id for v in voices[:8]])

    # Speak aloud with Kokoro
    print("\nSpeaking with Kokoro-82M ('af_heart')...")
    tts.speak("Hello! This is Kokoro-82M running locally with state-of-the-art neural speech synthesis.")

    # Speak with male voice 'am_adam'
    print("Switching voice to 'am_adam'...")
    tts.set_voice("am_adam")
    tts.speak("And this is Adam, speaking with high-fidelity American English neural voice.")

    # Save Kokoro audio to file
    output_path = "kokoro_sample.wav"
    saved = tts.save("Exporting Kokoro-82M neural speech directly to a 24 Kilohertz WAV file.", output_path)
    print(f"Saved audio to: {saved}")

    # Stream Kokoro audio byte chunks
    print("Streaming audio chunks...")
    chunks = list(tts.stream("Streaming text to byte chunks with Kokoro-82M.", chunk_size=4096))
    print(f"Streamed {len(chunks)} chunks ({sum(len(c) for c in chunks)} bytes).")


async def async_kokoro_demo():
    print("\n--- 2. Async Kokoro-82M Demo ---")
    tts = TTS(engine="kokoro", voice="af_bella")

    print("Async speaking with 'af_bella'...")
    await tts.speak_async("Asynchronous speech synthesis powered by Kokoro-82M.")

    print("Async file saving...")
    await tts.save_async("Async output save file", "kokoro_async.wav")
    print("Saved kokoro_async.wav!")


if __name__ == "__main__":
    kokoro_demo()
    asyncio.run(async_kokoro_demo())
