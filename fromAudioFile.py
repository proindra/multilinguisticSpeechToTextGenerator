
from faster_whisper import WhisperModel

model_size = "medium.en"

model = WhisperModel(
    model_size,
    device="cpu",
    compute_type="int8"
)

segments, info = model.transcribe("audio.mp3", beam_size=5)

for segment in segments:
    print(segment.text)
