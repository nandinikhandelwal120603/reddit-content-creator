#!/usr/bin/env python3
"""
Reddit Stories Audio Reader Pipeline
Converts scraped Reddit stories JSON into audio narrations (WAV files)
using local Kokoro-TTS. Skips video compilation to generate high-quality voiceovers directly.
"""

import os
import sys
import json
import re
import subprocess
import numpy as np
import soundfile as sf

# Voice configuration mapping using ONLY am_adam and af_heart at 1.5x speed
VOICE_MAP = {
    "female": ("af_heart", 1.50),
    "male":   ("am_adam",  1.50),
    "neutral":("af_heart", 1.50)
}


def detect_narrator_gender(text):
    """
    Scans the story text for age/gender markers like (f22), (29M), etc., to identify the narrator's gender.
    """
    female_patterns = [
        r'\bI\s*\(\s*[fF]\s*\d+\s*\)',
        r'\bI\s*\(\s*\d+\s*[fF]\s*\)',
        r'\bI\s*,\s*\d+[fF]\b',
        r'\bI\s*,\s*[fF]\d+\b',
        r'\bI\s+am\s+a\s+\d+\s*years?\s*old\s*female\b',
        r'\bI\s*\(female\s*\d+\s*\)',
        r'\bI\s*\(\s*\d+\s*female\s*\)',
    ]
    male_patterns = [
        r'\bI\s*\(\s*[mM]\s*\d+\s*\)',
        r'\bI\s*\(\s*\d+\s*[mM]\s*\)',
        r'\bI\s*,\s*\d+[mM]\b',
        r'\bI\s*,\s*[mM]\d+\b',
        r'\bI\s+am\s+a\s+\d+\s*years?\s*old\s*male\b',
        r'\bI\s*\(male\s*\d+\s*\)',
        r'\bI\s*\(\s*\d+\s*male\s*\)',
    ]
    
    for pattern in female_patterns:
        if re.search(pattern, text):
            return "female"
            
    for pattern in male_patterns:
        if re.search(pattern, text):
            return "male"
            
    return "neutral"


def select_narrator_voice(story_body):
    """
    Detects narrator gender and maps to voice: af_heart for female, am_adam for male.
    Both speak at 1.5x speed.
    """
    gender = detect_narrator_gender(story_body)
    print(f"Detected narrator gender: {gender}")
    voice, speed = VOICE_MAP.get(gender, VOICE_MAP["neutral"])
    print(f"Selected Voice: {voice}, Speed: {speed}")
    return voice, speed


def clean_text_for_tts(text):
    """
    Preprocesses text to clean up abbreviations and slang for natural voice reading.
    """
    # AITA / AITAH / AITH
    text = re.sub(r'\bAITAH\b', 'Am I the asshole', text, flags=re.IGNORECASE)
    text = re.sub(r'\bAITA\b', 'Am I the asshole', text, flags=re.IGNORECASE)
    text = re.sub(r'\bAITH\b', 'Am I the asshole', text, flags=re.IGNORECASE)
    
    # Replace (f22), (22f), (m29), (29m) etc.
    text = re.sub(r'\([fF](\d+)\)', r'female \1', text)
    text = re.sub(r'\([mM](\d+)\)', r'male \1', text)
    text = re.sub(r'\((\d+)[fF]\)', r'female \1', text)
    text = re.sub(r'\((\d+)[mM]\)', r'male \1', text)
    
    text = re.sub(r'\b[fF](\d+)\b', r'female \1', text)
    text = re.sub(r'\b[mM](\d+)\b', r'male \1', text)
    text = re.sub(r'\b(\d+)[fF]\b', r'female \1', text)
    text = re.sub(r'\b(\d+)[mM]\b', r'male \1', text)
    
    # Common abbreviations
    text = re.sub(r'\big\b', 'I guess', text, flags=re.IGNORECASE)
    text = re.sub(r'\bbc\b', 'because', text, flags=re.IGNORECASE)
    text = re.sub(r'\bWTH\b', 'what the hell', text, flags=re.IGNORECASE)
    text = re.sub(r'\bMIL\b', 'mother in law', text, flags=re.IGNORECASE)
    text = re.sub(r'\bFIL\b', 'father in law', text, flags=re.IGNORECASE)
    text = re.sub(r'\bSIL\b', 'sister in law', text, flags=re.IGNORECASE)
    text = re.sub(r'\bBIL\b', 'brother in law', text, flags=re.IGNORECASE)
    
    # u/username -> user username
    text = re.sub(r'\bu/([a-zA-Z0-9_-]+)\b', r'user \1', text)
    # r/subreddit -> subreddit subredditname
    text = re.sub(r'\br/([a-zA-Z0-9_-]+)\b', r'subreddit \1', text)
    
    # Clean symbols
    text = text.replace('&', ' and ')
    
    return text


def trim_silence_numpy(audio_array, threshold=0.003, padding_ratio=0.05):
    """
    Trims silent boundaries from the beginning and end of a numpy audio array.
    """
    padding = int(24000 * padding_ratio)
    active_indices = np.where(np.abs(audio_array) > threshold)[0]
    if len(active_indices) == 0:
        return np.array([], dtype=np.float32)
    start_idx = max(0, active_indices[0] - padding)
    end_idx = min(len(audio_array), active_indices[-1] + padding)
    return audio_array[start_idx:end_idx]


def generate_tts_audio(text, voice, speed, output_path):
    """
    Generates TTS audio file for text using Kokoro pipeline.
    Silent sections are trimmed, and speaker volume is boosted by 1.8x.
    """
    print(f"Generating TTS audio using voice={voice}, speed={speed}...")
    from kokoro import KPipeline
    
    lang = 'b' if voice.startswith('b') else 'a'
    pipeline = KPipeline(lang_code=lang)
    
    generator = pipeline(text, voice=voice, speed=speed, split_pattern=r'\n+|[.!?]+')
    
    all_audio = []
    for gs, ps, audio in generator:
        if audio is not None and len(audio) > 0:
            trimmed = trim_silence_numpy(audio)
            if len(trimmed) > 0:
                all_audio.append(trimmed)
                # Append a micro pause of silence (0.05 seconds = 1200 samples)
                all_audio.append(np.zeros(1200, dtype=np.float32))
            
    if not all_audio:
        raise ValueError("Kokoro TTS audio generation returned empty audio arrays.")
        
    full_audio = np.concatenate(all_audio)
    # Boost volume of narrator speaker by 1.8x (+5dB), clipping to prevent distortion
    full_audio = np.clip(full_audio * 1.8, -1.0, 1.0)
    
    sf.write(output_path, full_audio, 24000)
    print(f"TTS audio generated at {output_path}")
    return output_path


def process_story(story_data, index, output_dir="output_audio"):
    """
    Main processing loop for a single story.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    title = story_data["title"]
    story_body = story_data["story"]
    
    print(f"\n--- Processing Story {index}: '{title}' ---")
    
    # Step 1: Detect gender and map voice (Heart or Adam only)
    voice, speed = select_narrator_voice(story_body)
    
    # Create temp directory for separate files
    temp_dir = os.path.join(output_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    title_audio_path = os.path.join(temp_dir, "title_audio.wav")
    story_audio_path = os.path.join(temp_dir, "story_audio.wav")
    
    # Clean texts
    cleaned_title = clean_text_for_tts(title)
    cleaned_story_body = clean_text_for_tts(story_body)
    
    # Step 2: Generate TTS audios
    generate_tts_audio(cleaned_title, voice, speed, title_audio_path)
    generate_tts_audio(cleaned_story_body, voice, speed, story_audio_path)
    
    # Step 3: Stitch them together with a small pause (0.5s of silence) between title and story
    safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '_', '-')).rstrip()
    safe_title = safe_title[:40].replace(' ', '_')
    output_filename = f"story_{index}_{safe_title}.wav"
    final_output_path = os.path.join(output_dir, output_filename)
    
    # We will use ffmpeg filter to concatenate with a 0.8 second pause in between
    # 0.8s of silence at 24000Hz mono = 19200 samples of 0 value
    pause_path = os.path.join(temp_dir, "pause.wav")
    pause_samples = np.zeros(19200, dtype=np.int16)
    sf.write(pause_path, pause_samples, 24000)
    
    concat_cmd = [
        'ffmpeg', '-y',
        '-i', title_audio_path,
        '-i', pause_path,
        '-i', story_audio_path,
        '-filter_complex', '[0:a][1:a][2:a]concat=n=3:v=0:a=1[outa]',
        '-map', '[outa]',
        final_output_path
    ]
    subprocess.run(concat_cmd, check=True)
    
    # Clean up temp files
    try:
        for f in [title_audio_path, pause_path, story_audio_path]:
            if os.path.exists(f):
                os.remove(f)
        os.rmdir(temp_dir)
    except Exception as e:
        print(f"Warning: could not delete temp files: {e}")
        
    print(f"--- Finished Story Processing. Final audio: {final_output_path} ---\n")
    return final_output_path


def main():
    # Find input story file
    stories_file = "output/reddit_AITH_10.json"
    if not os.path.exists(stories_file):
        os.makedirs("output", exist_ok=True)
        json_files = [os.path.join("output", f) for f in os.listdir("output") if f.endswith(".json")]
        if json_files:
            stories_file = json_files[0]
        else:
            print("Error: No stories JSON files found in output/. Please scrape some stories first.")
            sys.exit(1)
            
    print(f"Loading stories from {stories_file}...")
    with open(stories_file, "r") as f:
        stories = json.load(f)
        
    if not stories:
        print("Error: Stories file is empty.")
        sys.exit(1)
        
    # Process the first 3 stories
    num_to_process = min(3, len(stories))
    print(f"Processing the first {num_to_process} stories for narration audios...")
    for idx in range(num_to_process):
        story = stories[idx]
        try:
            process_story(story, idx + 1)
        except Exception as e:
            print(f"Error processing story index {idx+1}: {e}")


if __name__ == "__main__":
    main()
