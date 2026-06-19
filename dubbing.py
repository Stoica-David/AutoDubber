import os
import sys
import json
import logging
import tempfile
import subprocess
import re
import argparse  # Added to handle flags like --log
from typing import List

import yt_dlp
import whisper
from deep_translator import GoogleTranslator
from gtts import gTTS

# --- AZURE-READY LOGGING BASE CONFIGURATION ---
logger = logging.getLogger("AutoDubber")
logger.setLevel(logging.INFO)

# StreamHandler always runs (natively captured by Azure App Insights / Container Logs)
stream_handler = logging.StreamHandler(sys.stdout)
stream_formatter = logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[stream_handler]
)


def extract_video_id(youtube_url: str) -> str | None:
    """Extract the ID of the youtube video."""
    patterns = [
        r"(?:v=)([0-9A-Za-z_-]{11})",
        r"(?:youtube\.com/)([0-9A-Za-z_-]{11})",
        r"(?:youtu\.be/)([0-9A-Za-z_-]{11})",
    ]
    for pat in patterns:
        m = re.search(pat, youtube_url)
        if m:
            return m.group(1)
    return None


def download_video(youtube_url: str, output_dir: str, base_name: str) -> str:
    """Download video from youtube into specified directory with specified name."""
    outtmpl = os.path.join(output_dir, f"{base_name}.%(ext)s")
    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": outtmpl,
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True
    }
    logger.info(f"Downloading video from YouTube to {output_dir}: {youtube_url}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    
    return os.path.join(output_dir, f"{base_name}.mp4")


def extract_audio_from_video(video_path: str, output_audio: str) -> str:
    """Extract audio path using ffmpeg."""
    logger.info(f"Extracting audio from: {video_path}")
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        output_audio
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_audio


def transcribe_audio(audio_path: str, source_lang: str, model_size: str = "base") -> str:
    """Transcribe audio path with Whisper based on source language from config file."""
    logger.info(f"Initializing Whisper ({model_size}) for language: {source_lang}")
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path, language=source_lang)
    logger.info("Transcription was successful.")
    return result["text"]


def split_text_into_digestable_chunks(text: str, max_chunk_length: int = 4999) -> List[str]:
    phrases = text.split(".")
    chunks = [phrases[0]]

    for phrase in phrases[1:]:
        current_phrase_length = len(phrase)
        if (len(chunks[-1]) + current_phrase_length) < max_chunk_length:
            chunks[-1] += "." + phrase
        else:
            chunks.append(phrase)
    return chunks


def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    """Translate the text from source language to target language."""
    logger.info(f"Translating text from [{source_lang}] to [{target_lang}]...")
    translated_transcript = ""
    transcript_chunks = split_text_into_digestable_chunks(text)

    translator = GoogleTranslator(source=source_lang, target=target_lang)
    for chunk in transcript_chunks:
        if chunk.strip():
            translated_transcript += translator.translate(chunk) + " "
    
    return translated_transcript.strip()


def generate_speech(text: str, output_speech: str, target_lang: str) -> str:
    """Generate target language speech with gTTS."""
    logger.info(f"Generating gTTS speech for language: {target_lang}")
    tts = gTTS(text=text, lang=target_lang)
    tts.save(output_speech)
    return output_speech


def merge_video_with_audio(video_path: str, audio_path: str, output_video: str) -> str:
    """Synchronize new audio file with original video."""
    logger.info(f"Merging final video: {output_video}")
    cmd = [
        "ffmpeg", "-i", video_path, "-i", audio_path,
        "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0",
        "-shortest", "-y", output_video
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_video


def process_single_video(video_config: dict) -> str:
    """Process singular element from the config file."""
    base_name = video_config["base_name"]
    url = video_config["yt_video_link"]
    src_lang = video_config.get("input_language", "en")
    tgt_lang = video_config.get("output_language", "ro")
    model_size = video_config.get("whisper_model", "base")
    clean_up = video_config.get("clean_temporary_files", True)

    logger.info(f"====== PROCESSING: {base_name.upper()} ======")

    src_dir = os.path.join("videos", src_lang)
    tgt_dir = os.path.join("videos", tgt_lang)
    
    os.makedirs(src_dir, exist_ok=True)
    os.makedirs(tgt_dir, exist_ok=True)
    
    final_output_path = os.path.join(tgt_dir, f"{base_name}.mp4")

    # Init temp files
    temp_dir = tempfile.gettempdir()
    orig_audio = os.path.join(temp_dir, f"{base_name}_orig_audio.wav")
    translated_audio = os.path.join(temp_dir, f"{base_name}_translated_audio.wav")
    video_path = None

    try:
        # 1. Source file identification
        if extract_video_id(url) is None:
            if os.path.isfile(url):
                video_path = url
                logger.info(f"Source was identified as local file: {video_path}")
            else:
                raise ValueError(f"The URL is not valid: {url}")
        else:
            video_path = download_video(url, src_dir, base_name)

        # 2. Processing pipeline
        extract_audio_from_video(video_path, orig_audio)
        raw_text = transcribe_audio(orig_audio, src_lang, model_size)
        translated_text = translate_text(raw_text, src_lang, tgt_lang)
        
        generate_speech(translated_text, translated_audio, tgt_lang)
        merge_video_with_audio(video_path, translated_audio, final_output_path)

        logger.info(f"The translated video was saved at: {final_output_path}")

    except Exception as e:
        logger.error(f"Error at processing '{base_name}': {str(e)}", exc_info=True)
        raise e

    finally:
        if clean_up:
            logger.info("Deleting temporary files...")
            for f in [orig_audio, translated_audio]:
                if os.path.exists(f):
                    os.remove(f)

    return final_output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoDubber Processing Pipeline")
    parser.add_argument(
        "--log", 
        action="store_true", 
        help="Save logs locally to pipeline.log file"
    )
    args = parser.parse_args()

    if args.log:
        file_handler = logging.FileHandler("pipeline.log", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s"))
        logger.addHandler(file_handler)
        logger.info("Local log saving enabled (--log flag recognized).")

    config_path = "config.json"
    
    if not os.path.exists(config_path):
        logger.critical(f"Config file '{config_path}' could not be found!")
        sys.exit(1)

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)
    except Exception as e:
        logger.critical(f"Error reading JSON file: {e}")
        sys.exit(1)

    video_list = config_data.get("videos", [])
    logger.info(f"{len(video_list)} videos are going to be processed.")

    successful_jobs = 0
    for idx, video_entry in enumerate(video_list):
        try:
            process_single_video(video_entry)
            successful_jobs += 1
        except Exception:
            logger.warning(f"The task no. {idx + 1} failed, skipping to next job in queue.")

    logger.info(f"Pipeline completed. Successful jobs: {successful_jobs}/{len(video_list)}")
