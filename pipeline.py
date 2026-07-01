#!/usr/bin/env python3
"""
Reddit Stories Video Production Pipeline
Converts scraped Reddit stories JSON into captioned vertical (9:16) videos
using local Kokoro-TTS, faster-whisper, and MoviePy.
"""

import os
import sys
import json
import re
import subprocess
import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont

# Page settings
PAGE_ID = "reddit.stories"

# Voice configuration mapping using ONLY am_adam and af_heart at 1.5x speed
VOICE_MAP = {
    "female": ("af_heart", 1.50),
    "male":   ("am_adam",  1.50),
    "neutral":("af_heart", 1.50)
}

STATE_FILE = "bg_video_state.json"


def get_bg_video_state():
    """Reads active background video index and playback offset from JSON file."""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                data = json.load(f)
                return int(data.get("index", 0)), float(data.get("position", 0.0))
        except:
            pass
    return 0, 0.0


def save_bg_video_state(index, position):
    """Saves updated background video index and offset to JSON state file."""
    with open(STATE_FILE, "w") as f:
        json.dump({"index": index, "position": position}, f)


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
    # AITA / AITAH / AITAH
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


def generate_title_card(title, subreddit="r/AITAH", upvotes="1.2k", comments="85", output_path="title_card.png"):
    """
    Generates a Reddit-style title card image in a transparent 1080x1920 canvas.
    Card is positioned at the top center of the canvas. Privacy preserved: uses PAGE_ID instead of author.
    """
    width, height = 1080, 1920
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    card_w = 920
    card_padding = 40
    
    # Font Selection
    try:
        font_path_bold = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        font_path_reg = "/System/Library/Fonts/Supplemental/Arial.ttf"
        if not os.path.exists(font_path_bold):
            font_path_bold = "/System/Library/Fonts/Helvetica.ttc"
            font_path_reg = "/System/Library/Fonts/Helvetica.ttc"
            
        font_subreddit = ImageFont.truetype(font_path_bold, 28)
        font_meta = ImageFont.truetype(font_path_reg, 22)
        font_title = ImageFont.truetype(font_path_bold, 36)
        font_stats = ImageFont.truetype(font_path_bold, 24)
    except IOError:
        font_subreddit = ImageFont.load_default()
        font_meta = ImageFont.load_default()
        font_title = ImageFont.load_default()
        font_stats = ImageFont.load_default()
        
    # Word wrap title text
    max_text_w = card_w - card_padding * 2
    words = title.split()
    lines = []
    current_line = []
    for word in words:
        test_line = ' '.join(current_line + [word])
        try:
            line_w = draw.textlength(test_line, font=font_title)
        except AttributeError:
            line_w = draw.textbbox((0, 0), test_line, font=font_title)[2]
        if line_w > max_text_w:
            if current_line:
                lines.append(' '.join(current_line))
                current_line = [word]
            else:
                lines.append(word)
                current_line = []
        else:
            current_line.append(word)
    if current_line:
        lines.append(' '.join(current_line))
        
    # Calculate title heights
    title_font_size = 36
    line_spacing = 10
    try:
        title_line_h = draw.textbbox((0, 0), "A", font=font_title)[3]
    except AttributeError:
        title_line_h = title_font_size
        
    title_total_h = len(lines) * title_line_h + (len(lines) - 1) * line_spacing
    card_h = 40 + 30 + title_total_h + 30 + 45 + 80
    
    # Position card at the top center of the screen
    card_x1 = (width - card_w) // 2
    card_y1 = 200 # positioned at y=200 on screen (blurred bg region)
    card_x2 = card_x1 + card_w
    card_y2 = card_y1 + card_h
    
    # Reddit dark mode: #1A1A1B, border #343536
    bg_color = (26, 26, 27, 255)
    border_color = (52, 53, 54, 255)
    
    # Draw card
    draw.rounded_rectangle([card_x1, card_y1, card_x2, card_y2], radius=20, fill=bg_color, outline=border_color, width=2)
    
    # Logo
    logo_cx = card_x1 + card_padding + 20
    logo_cy = card_y1 + card_padding + 20
    logo_r = 20
    draw.ellipse([logo_cx - logo_r, logo_cy - logo_r, logo_cx + logo_r, logo_cy + logo_r], fill=(255, 69, 0, 255))
    draw.text((logo_cx - 8, logo_cy - 12), "r/", fill=(255, 255, 255, 255), font=font_subreddit)
    
    # Subreddit Name
    sub_x = card_x1 + card_padding + 50
    sub_y = card_y1 + card_padding + 5
    draw.text((sub_x, sub_y), subreddit, fill=(215, 218, 220, 255), font=font_subreddit)
    
    # Metadata: Uses page ID instead of author name to protect privacy
    try:
        sub_w = draw.textlength(subreddit, font=font_subreddit)
    except AttributeError:
        sub_w = draw.textbbox((0, 0), subreddit, font=font_subreddit)[2]
        
    meta_text = f"• Posted by u/{PAGE_ID}"
    meta_x = sub_x + sub_w + 10
    meta_y = sub_y + 4
    draw.text((meta_x, meta_y), meta_text, fill=(129, 131, 132, 255), font=font_meta)
    
    # Title
    title_start_y = sub_y + 55
    for i, line in enumerate(lines):
        ly = title_start_y + i * (title_line_h + line_spacing)
        draw.text((card_x1 + card_padding, ly), line, fill=(215, 218, 220, 255), font=font_title)
        
    # Footer Stats
    footer_y = title_start_y + title_total_h + 25
    
    up_w, up_h = 130, 45
    up_x1 = card_x1 + card_padding
    up_y1 = footer_y
    up_x2 = up_x1 + up_w
    up_y2 = up_y1 + up_h
    draw.rounded_rectangle([up_x1, up_y1, up_x2, up_y2], radius=22, fill=(39, 39, 41, 255))
    draw.polygon([
        (up_x1 + 25, up_y1 + 15),
        (up_x1 + 15, up_y1 + 27),
        (up_x1 + 35, up_y1 + 27)
    ], fill=(129, 131, 132, 255))
    draw.text((up_x1 + 45, up_y1 + 8), str(upvotes), fill=(215, 218, 220, 255), font=font_stats)
    
    comm_w, comm_h = 150, 45
    comm_x1 = up_x2 + 20
    comm_y1 = footer_y
    comm_x2 = comm_x1 + comm_w
    comm_y2 = comm_y1 + comm_h
    draw.rounded_rectangle([comm_x1, comm_y1, comm_x2, comm_y2], radius=22, fill=(39, 39, 41, 255))
    draw.rounded_rectangle([comm_x1 + 18, comm_y1 + 14, comm_x1 + 38, comm_y1 + 28], radius=4, fill=(129, 131, 132, 255))
    draw.polygon([
        (comm_x1 + 22, comm_y1 + 27),
        (comm_x1 + 17, comm_y1 + 32),
        (comm_x1 + 26, comm_y1 + 27)
    ], fill=(129, 131, 132, 255))
    draw.text((comm_x1 + 48, comm_y1 + 8), str(comments), fill=(215, 218, 220, 255), font=font_stats)
    
    img.save(output_path, "PNG")
    print(f"Title card generated at {output_path}")
    return output_path


def generate_cta_card(output_path="cta_card.png"):
    """
    Generates a themed 'To Be Continued' card for video splits.
    """
    width, height = 1080, 1920
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    card_w = 900
    card_h = 350
    
    try:
        font_path_bold = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        font_path_reg = "/System/Library/Fonts/Supplemental/Arial.ttf"
        if not os.path.exists(font_path_bold):
            font_path_bold = "/System/Library/Fonts/Helvetica.ttc"
            font_path_reg = "/System/Library/Fonts/Helvetica.ttc"
            
        font_header = ImageFont.truetype(font_path_bold, 48)
        font_sub = ImageFont.truetype(font_path_reg, 32)
    except IOError:
        font_header = ImageFont.load_default()
        font_sub = ImageFont.load_default()
        
    card_x1 = (width - card_w) // 2
    card_y1 = (height - card_h) // 2
    card_x2 = card_x1 + card_w
    card_y2 = card_y1 + card_h
    
    bg_color = (26, 26, 27, 255)
    border_color = (255, 69, 0, 255)
    
    draw.rounded_rectangle([card_x1, card_y1, card_x2, card_y2], radius=25, fill=bg_color, outline=border_color, width=4)
    draw.text((width // 2, card_y1 + 80), "TO BE CONTINUED...", fill=(255, 69, 0, 255), font=font_header, anchor="mm")
    draw.text((width // 2, card_y1 + 200), "Check the channel for the full story!", fill=(215, 218, 220, 255), font=font_sub, anchor="mm")
    
    img.save(output_path, "PNG")
    print(f"CTA card generated at {output_path}")
    return output_path


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


def get_video_duration(path):
    """
    Returns duration of video using ffprobe.
    """
    probe = subprocess.run([
        'ffprobe', '-v', 'quiet', '-print_format', 'json',
        '-show_streams', path
    ], capture_output=True, text=True)
    info = json.loads(probe.stdout)
    video_stream = next(s for s in info['streams'] if s['codec_type'] == 'video')
    return float(video_stream['duration'])


def prepare_cropped_background(bg_video_path, output_path, duration, start_time, target_w=1080, target_h=1920):
    """
    Slices background video from start_time, speeds it up by 2.0x, crops to 9:16 layout without stretching.
    """
    print(f"Cropping and speeding up background video {bg_video_path} from offset {start_time:.2f}s...")
    
    speed_factor = 2.0
    input_slice_duration = duration * speed_factor
    
    vf = (
        f'[0:v]setpts=PTS/{speed_factor},scale={target_w}:{target_h}:force_original_aspect_ratio=increase,crop={target_w}:{target_h},boxblur=20:5[bg];'
        f'[0:v]setpts=PTS/{speed_factor},scale={target_w}:-2[fg];'
        f'[bg][fg]overlay=(W-w)/2:(H-h)/2[outv]'
    )
    
    cmd = [
        'ffmpeg',
        '-ss', f'{start_time:.2f}',
        '-t', f'{input_slice_duration:.2f}',
        '-i', bg_video_path,
        '-filter_complex', vf,
        '-map', '[outv]',
        '-an', # disable background audio (muted)
        '-c:v', 'libx264',
        '-crf', '18',
        '-preset', 'fast',
        '-y',
        output_path
    ]
    subprocess.run(cmd, check=True)
    print(f"Processed background clip saved to {output_path}")
    return output_path


def transcribe_audio_whisper(audio_path):
    """
    Transcribes audio using faster-whisper to extract word-level timestamps.
    """
    print("Transcribing audio for word-level timestamps...")
    from faster_whisper import WhisperModel
    
    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, info = model.transcribe(audio_path, word_timestamps=True)
    
    words_list = []
    for segment in segments:
        for word in segment.words:
            words_list.append({
                "word": word.word.strip(),
                "start": word.start,
                "end": word.end
            })
            
    print(f"Transcription finished. Transcribed {len(words_list)} words.")
    return words_list


def chunk_words(words, max_chars=18, max_words=3):
    """
    Groups individual word timestamps into short phrases (captions) of 2-3 words.
    Ensures that captions do not overlap in time.
    """
    chunks = []
    current_chunk_words = []
    
    for word_info in words:
        current_chunk_words.append(word_info)
        
        # Calculate properties of current group
        chunk_text = " ".join(w["word"] for w in current_chunk_words)
        
        # Check if we should close the chunk
        if len(current_chunk_words) >= max_words or len(chunk_text) >= max_chars:
            chunks.append({
                "text": chunk_text.upper(),
                "start": current_chunk_words[0]["start"],
                "end": current_chunk_words[-1]["end"]
            })
            current_chunk_words = []
            
    # Add any remaining words
    if current_chunk_words:
        chunks.append({
            "text": " ".join(w["word"] for w in current_chunk_words).upper(),
            "start": current_chunk_words[0]["start"],
            "end": current_chunk_words[-1]["end"]
        })
        
    # Clean up end times to prevent any overlap
    for i in range(len(chunks) - 1):
        chunks[i]["end"] = min(chunks[i]["end"], chunks[i+1]["start"])
        if chunks[i]["end"] <= chunks[i]["start"]:
            chunks[i]["end"] = chunks[i]["start"] + 0.2
            
    return chunks


def assemble_final_video(background_video, tts_audio, title_card_image, title_duration, words, show_cta_card, cta_card_image, output_path):
    """
    Composites everything using MoviePy: crops background, aligns audio, adds subtitle burns.
    """
    print("Assembling final video using MoviePy...")
    from moviepy import VideoFileClip, AudioFileClip, ImageClip, TextClip, CompositeVideoClip, vfx
    
    # Resolve absolute system font file path for Pillow rendering
    font_path = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
    if not os.path.exists(font_path):
        font_path = "/System/Library/Fonts/Helvetica.ttc"
    
    # Load background video and TTS audio
    video_clip = VideoFileClip(background_video)
    audio_clip = AudioFileClip(tts_audio)
    
    # Set video audio to TTS
    video_clip = video_clip.with_audio(audio_clip)
    
    total_duration = audio_clip.duration
    
    # Add title card overlay at start, fading out
    title_clip = (ImageClip(title_card_image)
                  .with_start(0)
                  .with_duration(title_duration)
                  .with_effects([vfx.FadeOut(0.4)]))
                  
    caption_clips = []
    
    # Group words into clean caption chunks (2-3 words) to fix overlapping and clutter
    caption_chunks = chunk_words(words, max_chars=18, max_words=3)
    
    # Create word-by-word caption clips
    for idx, chunk in enumerate(caption_chunks):
        if chunk["start"] < title_duration:
            continue
            
        if show_cta_card and chunk["start"] >= total_duration - 4.0:
            continue
            
        chunk_text = chunk["text"]
        start_t = chunk["start"]
        end_t = chunk["end"]
            
        if show_cta_card and end_t >= total_duration - 4.0:
            end_t = total_duration - 4.0
            
        if end_t <= start_t:
            continue
            
        txt_clip = TextClip(
            text=chunk_text,
            font_size=90,
            font=font_path,
            color="yellow",
            stroke_color="black",
            stroke_width=4,
            method="caption",
            size=(950, None),
            margin=(15, 15)
        ).with_position(("center", "center")) \
         .with_start(start_t) \
         .with_end(end_t)
         
        caption_clips.append(txt_clip)
        
    # Watermark overlay
    watermark_img_path = "ChatGPT_Image_Jul_1__2026__07_27_20_PM-removebg-preview.png"
    if os.path.exists(watermark_img_path):
        watermark_clip = (ImageClip(watermark_img_path)
                          .with_effects([vfx.Resize(width=130), vfx.Rotate(15, expand=True)])
                          .with_start(0)
                          .with_duration(total_duration)
                          .with_opacity(0.6)
                          .with_position((880, 70)))
    else:
        # Fallback to TextClip watermark if image not found
        watermark_clip = TextClip(
            text=f"@{PAGE_ID}",
            font_size=28,
            font=font_path,
            color="white",
            stroke_color="black",
            stroke_width=1,
            method="caption",
            size=(200, None)
        ).with_position((860, 45)) \
         .with_start(0) \
         .with_duration(total_duration) \
         .with_opacity(0.5)
     
    clips_list = [video_clip, title_clip, watermark_clip] + caption_clips
    
    # If CTA card needs to show up
    if show_cta_card and cta_card_image:
        cta_clip = (ImageClip(cta_card_image)
                    .with_start(total_duration - 4.0)
                    .with_duration(4.0)
                    .with_effects([vfx.FadeIn(0.4)]))
        clips_list.append(cta_clip)
        
    final_clip = CompositeVideoClip(clips_list)
    
    # Write output file
    final_clip.write_videofile(
        output_path,
        fps=30,
        codec="libx264",
        audio_codec="aac",
        threads=4
    )
    
    # Clean up file handles
    video_clip.close()
    audio_clip.close()
    title_clip.close()
    watermark_clip.close()
    for clip in caption_clips:
        clip.close()
    if show_cta_card:
        cta_clip.close()
    final_clip.close()
    
    print(f"Final video generated at {output_path}")
    return output_path


def process_story(story_data, bg_video_list, story_index, output_dir="output_shorts"):
    """
    Main processing loop for a single story.
    Handles gender mappings, Kokoro narration, video slicing/speedup, and CTA cap.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    title = story_data["title"]
    upvotes = story_data.get("upvotes", "1.0k")
    comments = story_data.get("comments", "50")
    story_body = story_data["story"]
    
    print(f"\n--- Processing Story {story_index}: '{title}' ---")
    
    # Step 1: Detect gender and map voice (Heart or Adam only)
    voice, speed = select_narrator_voice(story_body)
    
    # Create temp directory
    temp_dir = os.path.join(output_dir, f"temp_{story_index}")
    os.makedirs(temp_dir, exist_ok=True)
    
    title_audio_path = os.path.join(temp_dir, "title_audio.wav")
    story_audio_path = os.path.join(temp_dir, "story_audio.wav")
    full_audio_path = os.path.join(temp_dir, "full_audio.wav")
    
    # Step 2: Generate TTS audio
    cleaned_title = clean_text_for_tts(title)
    cleaned_story_body = clean_text_for_tts(story_body)
    
    # Generate separate title and story audio to measure title duration
    generate_tts_audio(cleaned_title, voice, speed, title_audio_path)
    generate_tts_audio(cleaned_story_body, voice, speed, story_audio_path)
    
    # Find duration of title audio
    title_info = sf.info(title_audio_path)
    title_duration = title_info.duration
    print(f"Title audio duration: {title_duration:.2f} seconds")
    
    # Concatenate title + small pause (0.5s) + story body using ffmpeg
    pause_path = os.path.join(temp_dir, "pause.wav")
    pause_samples = np.zeros(12000, dtype=np.int16) # 0.5s of silence at 24000Hz mono
    sf.write(pause_path, pause_samples, 24000)
    
    concat_cmd = [
        'ffmpeg', '-y',
        '-i', title_audio_path,
        '-i', pause_path,
        '-i', story_audio_path,
        '-filter_complex', '[0:a][1:a][2:a]concat=n=3:v=0:a=1[outa]',
        '-map', '[outa]',
        full_audio_path
    ]
    subprocess.run(concat_cmd, check=True)
    
    total_audio_duration = sf.info(full_audio_path).duration
    print(f"Initial total audio duration: {total_audio_duration:.2f} seconds")
    
    # Step 3: Handle 3-Minute (180s) limit cap
    show_cta_card = False
    cta_card_path = None
    if total_audio_duration > 180.0:
        print("Story duration exceeds 3 minutes! Truncating and adding channel CTA...")
        # Slice the main audio to 176.0s
        audio_data, sr = sf.read(full_audio_path)
        sliced_samples = audio_data[:int(176.0 * sr)]
        
        # Generate CTA narration audio using the same voice and speed
        cta_audio_path = os.path.join(temp_dir, "cta_audio.wav")
        generate_tts_audio("Check the channel for the full video.", voice, speed, cta_audio_path)
        cta_data, _ = sf.read(cta_audio_path)
        
        # Combine
        combined_audio = np.concatenate([sliced_samples, cta_data])
        sf.write(full_audio_path, combined_audio, sr)
        
        # Generate CTA card
        cta_card_path = os.path.join(temp_dir, "cta_card.png")
        generate_cta_card(cta_card_path)
        
        show_cta_card = True
        total_audio_duration = sf.info(full_audio_path).duration
        print(f"Adjusted total audio duration with CTA: {total_audio_duration:.2f} seconds")
        
    # Step 4: Draw Title Card (Privacy mode - u/reddit.stories)
    title_card_path = os.path.join(temp_dir, "title_card.png")
    generate_title_card(title, subreddit="r/AITAH", upvotes=upvotes, comments=comments, output_path=title_card_path)
    
    # Step 5: Background video slicing (continuous offset tracking)
    bg_idx, bg_start_pos = get_bg_video_state()
    current_bg_video = bg_video_list[bg_idx % len(bg_video_list)]
    bg_duration = get_video_duration(current_bg_video)
    
    # Required background video input length is total_audio_duration * 2 (since played at 2x speed)
    req_input_len = total_audio_duration * 2.0
    
    if bg_start_pos + req_input_len > bg_duration - 10:
        # Switch to the next video in list
        bg_idx = (bg_idx + 1) % len(bg_video_list)
        bg_start_pos = 0.0
        current_bg_video = bg_video_list[bg_idx]
        bg_duration = get_video_duration(current_bg_video)
        print(f"Reached end of background video. Switching to next base: {current_bg_video} at 0.0s.")
        
    cropped_bg_path = os.path.join(temp_dir, "cropped_bg.mp4")
    prepare_cropped_background(
        bg_video_path=current_bg_video,
        output_path=cropped_bg_path,
        duration=total_audio_duration,
        start_time=bg_start_pos,
        target_w=1080,
        target_h=1920
    )
    
    # Update and save the background position offset for next story
    next_bg_pos = bg_start_pos + req_input_len
    save_bg_video_state(bg_idx, next_bg_pos)
    print(f"Updated background video index to {bg_idx} and offset to {next_bg_pos:.2f}s")
    
    # Step 6: Transcribe Voiceover using Whisper
    words_list = transcribe_audio_whisper(full_audio_path)
    
    # Step 7: Assemble final clip
    safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '_', '-')).rstrip()
    safe_title = safe_title[:40].replace(' ', '_')
    output_filename = f"short_{story_index}_{safe_title}.mp4"
    final_output_path = os.path.join(output_dir, output_filename)
    
    assemble_final_video(
        background_video=cropped_bg_path,
        tts_audio=full_audio_path,
        title_card_image=title_card_path,
        title_duration=title_duration,
        words=words_list,
        show_cta_card=show_cta_card,
        cta_card_image=cta_card_path,
        output_path=final_output_path
    )
    
    # Clean up temp files
    try:
        for f in [title_audio_path, story_audio_path, pause_path, full_audio_path, title_card_path, cropped_bg_path]:
            if f and os.path.exists(f):
                os.remove(f)
        if show_cta_card:
            if os.path.exists(cta_card_path):
                os.remove(cta_card_path)
            if os.path.exists(os.path.join(temp_dir, "cta_audio.wav")):
                os.remove(os.path.join(temp_dir, "cta_audio.wav"))
        os.rmdir(temp_dir)
    except Exception as e:
        print(f"Warning: could not delete temp files: {e}")
        
    print(f"--- Finished Story Processing. Final video: {final_output_path} ---\n")
    return final_output_path


def main():
    # Set background video list
    bg_video_list = ["background_videos/video_5.mp4", "background_videos/video_2.mp4"]
    for bg_path in bg_video_list:
        if not os.path.exists(bg_path):
            print(f"Error: Background video {bg_path} not found in workspace.")
            sys.exit(1)
        
    # Reset background position offset to 0.0 at the beginning of a fresh run
    save_bg_video_state(0, 0.0)
    
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
    print(f"Processing the first {num_to_process} stories to vertical captioned shorts...")
    for idx in range(num_to_process):
        story = stories[idx]
        print(f"\nProcessing story index {idx+1}/{num_to_process}...")
        try:
            process_story(story, bg_video_list, idx + 1)
        except Exception as e:
            print(f"Error processing story index {idx+1}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
