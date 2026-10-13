# Real-Time Speech Transcription with Faster-Whisper

> Built on top of [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — a reimplementation of OpenAI's Whisper using [CTranslate2](https://github.com/OpenNMT/CTranslate2), which is up to 4x faster than the original Whisper with lower memory usage.

Real-time microphone transcription and audio file transcription using [faster-whisper](https://github.com/SYSTRAN/faster-whisper) on CUDA GPU.

## Files

| File | Description |
|------|-------------|
| `realTimeTranscription.py` | Live microphone transcription using CUDA |
| `fromAudioFile.py` | Transcribe a local audio file using CPU |
| `requirements.txt` | Python dependencies |

## Requirements

- Python 3.11
- NVIDIA GPU with CUDA support
- [cuDNN 9.x](https://developer.nvidia.com/cudnn) installed on Windows
- Conda environment

## Setup

1. Create and activate the conda environment:
   ```bash
   conda create -p ./sevenv python=3.11
   conda activate ./sevenv
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Real-time microphone transcription
```bash
python realTimeTranscription.py
```
- Listens to your microphone and transcribes every 3 seconds
- Uses `large-v3-turbo` model on CUDA with VAD filtering
- Press `Ctrl+C` to stop

### Transcribe an audio file
```bash
python fromAudioFile.py
```
- Transcribes `audio.mp3` using `medium.en` model on CPU

## Configuration

In `realTimeTranscription.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `chunk_duration` | `3` | Seconds of audio per transcription chunk |
| `block_duration` | `0.5` | Microphone capture block size in seconds |
| `model` | `large-v3-turbo` | Whisper model size |
| `language` | `None` | Set to `"en"` to skip auto language detection |
| `beam_size` | `3` | Higher = more accurate, slower |
