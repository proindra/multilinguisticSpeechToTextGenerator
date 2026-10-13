# Real-Time Speech Transcription with Faster-Whisper

> Built on top of [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — a reimplementation of OpenAI's Whisper using [CTranslate2](https://github.com/OpenNMT/CTranslate2), which is up to 4x faster than the original Whisper with lower memory usage.

Real-time microphone transcription, audio file transcription, English-to-Hindi translation, and text-to-speech using [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and [edge-tts](https://github.com/rany2/edge-tts).

## Files

| File | Description |
|------|-------------|
| `realtime_transcription.py` | Live microphone transcription using CUDA |
| `realtime_translator.py` | Live English speech → Hindi text + speech using CUDA |
| `transcribe_audio_file.py` | Transcribe a local audio file using CPU |
| `text_to_speech.py` | Convert text from `input.txt` to speech using Microsoft Edge TTS |
| `input.txt` | Input text file for TTS (supports Hindi + English) |
| `realtime_translator.txt` | Auto-generated log of English + Hindi translations |
| `requirements.txt` | Python dependencies with pinned versions |

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
python realtime_transcription.py
```
- Listens to your microphone and transcribes every 3 seconds
- Uses `large-v3-turbo` model on CUDA with VAD filtering
- Auto-detects language
- Press `Ctrl+C` to stop

### Real-time English to Hindi translator
```bash
python realtime_translator.py
```
- Listens to your microphone, transcribes English speech every 4 seconds
- Translates to Hindi using [MyMemory](https://mymemory.translated.net/) in a background thread
- Speaks the Hindi translation aloud via Microsoft Edge TTS
- Logs all translations to `realtime_translator.txt`
- Press `Ctrl+C` to stop

### Transcribe an audio file
```bash
python transcribe_audio_file.py
```
- Transcribes `audio.mp3` using `medium.en` model on CPU

### Text to speech
```bash
python text_to_speech.py
```
- Reads text from `input.txt` (supports Hindi + English mixed text)
- Uses Microsoft Edge TTS with `hi-IN-SwaraNeural` Hindi voice
- Saves output to `speech.mp3`

## Configuration

### `realtime_transcription.py`

| Variable | Default | Description |
|----------|---------|-------------|
| `chunk_duration` | `3` | Seconds of audio per transcription chunk |
| `block_duration` | `0.5` | Microphone capture block size in seconds |
| `model` | `large-v3-turbo` | Whisper model size |
| `language` | `None` | Set to `"en"` to skip auto language detection |
| `beam_size` | `3` | Higher = more accurate, slower |

### `realtime_translator.py`

| Variable | Default | Description |
|----------|---------|-------------|
| `CHUNK_DURATION` | `4` | Seconds of audio per transcription chunk |
| `BLOCK_DURATION` | `0.5` | Microphone capture block size in seconds |
| `HINDI_VOICE` | `hi-IN-SwaraNeural` | Edge TTS Hindi voice |
| `beam_size` | `1` | Lower = faster transcription |

### `text_to_speech.py`

| Variable | Default | Description |
|----------|---------|-------------|
| `voice` | `hi-IN-SwaraNeural` | Edge TTS voice |
| `rate` | `+0%` | Speech rate adjustment |
| `output` | `speech.mp3` | Output audio file |

## File Naming Convention

All Python files use `snake_case` naming:

| Old Name | New Name |
|----------|----------|
| `realTimeTranscription.py` | `realtime_transcription.py` |
| `fromAudioFile.py` | `transcribe_audio_file.py` |
| `fromTextToSpeech.py` | `text_to_speech.py` |
