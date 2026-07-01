# reddit-content-creator

A Python-based video pipeline that automates the creation of captioned vertical shorts from Reddit stories. It handles text-to-speech generation, caption synchronization using transcription, and overlaying narration and titles onto background videos.

## Features

- **Reddit Stories Processing**: Parses stories and converts them into engaging visual content.
- **Text-to-Speech (TTS)**: Leverages Kokoro / TTS engines to generate natural sounding voiceovers.
- **Audio-Video Synchronization**: Transcribes audio and automatically places timed word-by-word captions on top of background videos.
- **Dynamic Formatting**: Generates title cards and custom watermarks for social media platforms.

## Prerequisites

### Dependencies
This project requires Python 3.10+ and `FFmpeg` installed on your system.

### Installing FFmpeg
- **Mac (via Homebrew)**:
  ```bash
  brew install ffmpeg
  ```
- **Windows (via Winget)**:
  ```cmd
  winget install Gyan.FFmpeg
  ```

## Local Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/nandinikhandelwal120603/reddit-content-creator.git
   cd reddit-content-creator
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `.\venv\Scripts\activate`
   ```

3. **Install python packages**:
   ```bash
   pip install --upgrade pip
   pip install torch soundfile numpy pillow moviepy faster-whisper kokoro
   ```

## Usage

1. Put your background videos into the `background_videos/` directory.
2. Put the Reddit stories JSON into the `output/` directory.
3. Run the automated video pipeline:
   ```bash
   python pipeline.py
   ```
4. Find the generated vertical shorts in the `output_shorts/` directory.
