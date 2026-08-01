"""
Unit and integration tests for the TTS Python library (including Kokoro-82M).
"""

import asyncio
import os
import tempfile
import unittest
from tts import TTS, Voice, KokoroBackend


class TestTTSLibrary(unittest.TestCase):

    def setUp(self):
        self.tts = TTS(engine="kokoro", voice="af_heart", rate=180, volume=0.9)

    def test_kokoro_backend_voices(self):
        voices = self.tts.list_voices()
        self.assertIsInstance(voices, list)
        self.assertGreater(len(voices), 10)
        voice_ids = [v.id for v in voices]
        self.assertIn("af_heart", voice_ids)
        self.assertIn("af_bella", voice_ids)
        self.assertIn("am_adam", voice_ids)

    def test_kokoro_save_wav(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            res = self.tts.save("Hello Kokoro-82M neural model test save", tmp_path, audio_format="wav")
            self.assertEqual(res, tmp_path)
            self.assertTrue(os.path.exists(tmp_path))
            self.assertGreater(os.path.getsize(tmp_path), 5000)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_kokoro_stream_chunks(self):
        chunks = list(self.tts.stream("Testing Kokoro streaming audio chunks", chunk_size=4096))
        self.assertGreater(len(chunks), 0)
        total_bytes = sum(len(c) for c in chunks)
        self.assertGreater(total_bytes, 5000)

    def test_async_kokoro(self):
        async def run_async_tests():
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                res = await self.tts.save_async("Testing async Kokoro synthesis", tmp_path)
                self.assertTrue(os.path.exists(tmp_path))
                self.assertGreater(os.path.getsize(tmp_path), 5000)

                chunks = []
                async for chunk in self.tts.stream_async("Testing async streaming", chunk_size=2048):
                    chunks.append(chunk)
                self.assertGreater(len(chunks), 0)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        asyncio.run(run_async_tests())


if __name__ == "__main__":
    unittest.main()
