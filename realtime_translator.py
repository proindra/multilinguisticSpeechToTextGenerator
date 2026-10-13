import os
import glob
import time
import asyncio
import queue
import tempfile
import threading
from pathlib import Path

import numpy as np
import sounddevice as sd
import pygame
import edge_tts
from deep_translator import MyMemoryTranslator

# --------------------------------------------------
# 1. Register NVIDIA DLL directories on Windows
#    Do this before importing Faster Whisper.
# --------------------------------------------------

_nvidia_bins = glob.glob(
    os.path.join(os.path.dirname(__file__), "sevenv", "Lib", "site-packages", "nvidia", "*", "bin")
)
if _nvidia_bins:
    os.environ["PATH"] = os.pathsep.join(_nvidia_bins) + os.pathsep + os.environ.get("PATH", "")

from faster_whisper import WhisperModel

# --------------------------------------------------
# 2. Settings
# --------------------------------------------------

SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_DURATION = 0.5
CHUNK_DURATION = 4

FRAMES_PER_BLOCK = int(SAMPLE_RATE * BLOCK_DURATION)
FRAMES_PER_CHUNK = int(SAMPLE_RATE * CHUNK_DURATION)

MODEL_NAME = "deepdml/faster-whisper-large-v3-turbo-ct2"
HINDI_VOICE = "hi-IN-SwaraNeural"

audio_queue = queue.Queue(maxsize=16)
speech_queue = queue.Queue()

LOG_FILE = Path(__file__).with_name("realtime_translator.txt")


def log(english, hindi):
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(f"English: {english}\nHindi:   {hindi}\n\n")



# --------------------------------------------------
# 3. Load your existing Faster Whisper model
# --------------------------------------------------

print("Loading Faster Whisper large-v3-turbo...")

model = WhisperModel(
    "large-v3-turbo",
    device="cuda",
    compute_type="float16"
)

print("Model loaded successfully.")

# --------------------------------------------------
# 4. Microphone callback
# --------------------------------------------------

def audio_callback(indata, frames, time_info, status):
    if status:
        print("Audio status:", status)

    try:
        audio_queue.put_nowait(indata.copy())
    except queue.Full:
        # Drop the oldest block rather than letting latency grow.
        try:
            audio_queue.get_nowait()
        except queue.Empty:
            pass

        try:
            audio_queue.put_nowait(indata.copy())
        except queue.Full:
            pass

# --------------------------------------------------
# 5. Text-to-speech worker
#    Generates Hindi speech online and plays it.
# --------------------------------------------------

async def generate_hindi_speech(text, filename):
    communicate = edge_tts.Communicate(
        text,
        voice=HINDI_VOICE,
        rate="+0%"
    )
    await communicate.save(filename)


def tts_worker():
    pygame.mixer.init()

    while True:
        hindi_text = speech_queue.get()

        if hindi_text is None:
            speech_queue.task_done()
            break

        filename = None

        try:
            with tempfile.NamedTemporaryFile(
                suffix=".mp3",
                delete=False
            ) as temp_file:
                filename = temp_file.name

            asyncio.run(
                generate_hindi_speech(hindi_text, filename)
            )

            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                time.sleep(0.1)

            pygame.mixer.music.unload()

        except Exception as exc:
            print(f"\n[TTS error] {exc}")

        finally:
            if filename and os.path.exists(filename):
                try:
                    os.remove(filename)
                except OSError:
                    pass

            speech_queue.task_done()

    pygame.mixer.quit()


def _translate_and_queue(english_text):
    try:
        hindi_text = MyMemoryTranslator(source="en", target="hi").translate(english_text)
        if not hindi_text:
            return
        print(f"Hindi:   {hindi_text}")
        log(english_text, hindi_text)
        speech_queue.put(hindi_text)
    except Exception as exc:
        print(f"\n[Translation error] {exc}")
        log(english_text, "[translation failed]")


# --------------------------------------------------
# 6. English speech -> text -> Hindi translation
# --------------------------------------------------

def translate_and_speak(audio_data):
    try:
        segments, info = model.transcribe(
            audio_data,
            language="en",
            task="transcribe",
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=False
        )

        english_text = " ".join(
            segment.text.strip()
            for segment in segments
            if segment.text.strip()
        ).strip()

        if not english_text:
            return

        print(f"\nEnglish: {english_text}")
        threading.Thread(target=_translate_and_queue, args=(english_text,), daemon=True).start()

    except Exception as exc:
        print(f"\n[Transcription error] {exc}")


# --------------------------------------------------
# 7. Main real-time loop
# --------------------------------------------------

def main():
    tts_thread = threading.Thread(
        target=tts_worker,
        daemon=True
    )
    tts_thread.start()

    print("\nEnglish-to-Hindi translator is ready.")
    print("Speak English into your microphone.")
    print("Press Ctrl+C to stop.\n")

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=FRAMES_PER_BLOCK,
            callback=audio_callback
        ):
            while True:
                blocks = []
                collected_frames = 0

                # Collect approximately four seconds of audio.
                while collected_frames < FRAMES_PER_CHUNK:
                    block = audio_queue.get()
                    blocks.append(block)
                    collected_frames += len(block)

                audio_data = np.concatenate(
                    blocks,
                    axis=0
                )[:FRAMES_PER_CHUNK]

                audio_data = audio_data.flatten().astype(
                    np.float32
                )

                translate_and_speak(audio_data)

    except KeyboardInterrupt:
        print("\nStopping translator...")

    finally:
        speech_queue.put(None)
        tts_thread.join()

        print("Translator stopped.")


if __name__ == "__main__":
    main()
