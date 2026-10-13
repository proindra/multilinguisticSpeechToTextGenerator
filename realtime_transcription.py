import os
import glob

# Add nvidia CUDA DLL paths so faster-whisper can find cublas64_12.dll etc.
_nvidia_bins = glob.glob(
    os.path.join(os.path.dirname(__file__), "sevenv", "Lib", "site-packages", "nvidia", "*", "bin")
)
if _nvidia_bins:
    os.environ["PATH"] = os.pathsep.join(_nvidia_bins) + os.pathsep + os.environ.get("PATH", "")

import sounddevice as sd
import numpy as np
import queue
import threading
from faster_whisper import WhisperModel

# Settings
samplerate = 16000
block_duration = 0.5  # seconds
chunk_duration = 3  # seconds
channels = 1

frames_per_block = int(samplerate * block_duration)
frames_per_chunk = int(samplerate * chunk_duration)

audio_queue = queue.Queue()
audio_buffer = []

# Model setup: large-v3 on CUDA
model = WhisperModel(
    # "medium.en",
    language="en",
    model_size="large-v3-turbo",
    device="cuda",
    compute_type="float16"
)

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())


def recorder():
    with sd.InputStream(
        samplerate=samplerate,
        channels=channels,
        callback=audio_callback,
        blocksize=frames_per_block
    ):
        print("Listening... Press Ctrl+C to stop.")
        while True:
            sd.sleep(100)


def transcriber():
    global audio_buffer

    while True:
        block = audio_queue.get()
        audio_buffer.append(block)

        total_frames = sum(len(b) for b in audio_buffer)

        if total_frames >= frames_per_chunk:
            audio_data = np.concatenate(audio_buffer, axis=0)[:frames_per_chunk]
            audio_buffer = []  # Clear buffer

            audio_data = audio_data.flatten().astype(np.float32)

            # Transcription without timestamps
            segments, info = model.transcribe(
                audio_data,
                # language="en",
                language=None,  # Automatic language detection
                beam_size=3,
                temperature=0.0,
                vad_filter=True,
                condition_on_previous_text=False
            )

            for segment in segments:
                print(segment.text)

# start threads for recording and transcribing
threading.Thread(target=recorder, daemon=True).start()
try:
    transcriber()
except KeyboardInterrupt:
    print("\nStopped.")

