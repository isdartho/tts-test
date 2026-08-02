"""
Example Usage of Dedicated Kokoro-82M Text-to-Speech Library (tts.py)
"""

import asyncio
from tts import TTS


def sync_demo():
    print("--- 1. Synchronous Speech & File Export Demo ---")
    tts = TTS(voice="af_heart", rate=180)

    # 1. List voices
    voices = tts.list_voices()
    print(f"Loaded {len(voices)} Kokoro neural voices.")
    print("Sample voices:", [v.id for v in voices[:6]])

    # 2. Speak aloud
    print("\nSpeaking with 'af_heart'...")
    tts.speak("Hello! This is a dedicated Kokoro-82M neural Text-to-Speech engine.")

    # 3. Save audio file
    saved = tts.save("Exporting Kokoro speech directly to audio file.", "kokoro_demo.wav")
    print(f"Saved audio to: {saved}")

    # 4. Stream audio chunks
    print("Streaming audio chunks...")
    chunks = list(tts.stream("Streaming text to byte chunks.", chunk_size=4096))
    print(f"Streamed {len(chunks)} chunks ({sum(len(c) for c in chunks)} bytes).")


async def async_demo():
    print("\n--- 2. Asynchronous (asyncio) Demo ---")
    tts = TTS(voice="am_adam")

    print("Async speaking with 'am_adam'...")
    await tts.speak_async("Asynchronous speech synthesis with Kokoro-82M.")

    print("Async file export...")
    await tts.save_async("Exporting async audio.", "kokoro_async_demo.wav")
    print("Async export completed!")


if __name__ == "__main__":
    sync_demo()
    asyncio.run(async_demo())
