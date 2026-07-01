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

## Local Setup (Mac/Linux)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/nandinikhandelwal120603/reddit-content-creator.git
   cd reddit-content-creator
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate
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

---

## Extracting Split Video Files

Since `video_2.mp4` is larger than GitHub's 100MB file limit, it has been compressed and split into parts inside `background_videos/`. To reconstruct the original file after cloning:

```bash
# 1. Combine and fix the split archive into a single zip file
zip -F background_videos/video_2_split.zip --out background_videos/video_2_combined.zip

# 2. Extract the combined zip file to get the original mp4
unzip background_videos/video_2_combined.zip

# 3. Clean up the combined zip file (optional)
rm background_videos/video_2_combined.zip
```

---

## Windows Setup Guide

Follow these step-by-step instructions to set up and run the Reddit Stories Video Pipeline on your Windows laptop/PC.

### 1. Prerequisites (What to Download)

#### A. Python (Version 3.10 or 3.11 recommended)
1. Download the Python installer from [python.org](https://www.python.org/downloads/).
2. Run the installer and **IMPORTANT**: Make sure to check the box that says **"Add Python.exe to PATH"** at the bottom of the installer window before clicking Install.

#### B. FFmpeg (For video and audio slicing/speedups)
FFmpeg is a command-line tool that must be installed on your system.
* **Option 1 (Easiest)**: Open your Windows Terminal (Command Prompt or PowerShell) and run:
  ```cmd
  winget install Gyan.FFmpeg
  ```
* **Option 2 (Manual)**:
  1. Download the latest release build from [ffmpeg.org](https://ffmpeg.org/download.html).
  2. Extract the folder (e.g., to `C:\ffmpeg`).
  3. Add `C:\ffmpeg\bin` to your system environment variable `PATH` (System Properties -> Environment Variables -> Edit PATH -> Add).

---

### 2. Transferring the Project Files
Copy the entire `reddit stories` folder from your Mac to your Windows laptop. Keep the directory structure exactly the same:
```text
reddit stories/
├── background_videos/
│   ├── video_2.mp4
│   └── video_5.mp4
├── output/
│   └── reddit_AITH_10.json
├── pipeline.py
├── read_stories.py
└── WINDOWS_SETUP.md
```
*(Do NOT copy the `venv` directory over, as Python virtual environments are platform-specific and must be regenerated on Windows).*

---

### 3. Installation Steps on Windows

Open **PowerShell** or **Command Prompt** in the `reddit stories` project folder:

#### Step 1: Create a fresh Virtual Environment
```powershell
python -m venv venv
```

#### Step 2: Activate the Virtual Environment
* **On PowerShell**:
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
  *(If you get a permission/execution policy error, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first in PowerShell).*
* **On Command Prompt (cmd)**:
  ```cmd
  .\venv\Scripts\activate.bat
  ```

#### Step 3: Upgrade pip and install libraries
Run the following commands inside the activated environment:
```cmd
pip install --upgrade pip
pip install torch soundfile numpy pillow moviepy faster-whisper kokoro
```

---

### 4. How to Run the Scripts

Ensure your background videos are inside the `background_videos/` folder, and stories JSON is inside `output/`.

#### Run Video Production Pipeline:
```cmd
python pipeline.py
```
This will compile the first 3 stories into captioned vertical shorts under the `output_shorts/` directory, using `video_5.mp4` and rotating to `video_2.mp4` if it runs out of time.

#### Run Audio-Only Narration (Optional):
```cmd
python read_stories.py
```
This will generate plain WAV voiceovers (without video) under the `output_audio/` directory.

---

### Key Pipeline Settings (Customizable)

Inside `pipeline.py`, you can customize these top-level parameters:
* `PAGE_ID = "reddit.stories"`: Changes the page handle/author name displayed on the title card.
* `WATERMARK_X = 860` & `WATERMARK_Y = 45`: Changes the position of your watermarking handle.
