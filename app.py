from faster_whisper import WhisperModel
import edge_tts
import asyncio
from deep_translator import GoogleTranslator
import gradio as gr
import os
import tempfile
import atexit

import soundfile as sf
import numpy as np
import noisereduce as nr
import torch

# -------------------------------
# Load Models (GPU float16 for speed)
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"

print(f"[INFO] Loading Whisper on: {device.upper()} ({compute_type})")
model = WhisperModel("small", device=device, compute_type=compute_type)
print("[INFO] Model loaded ✓")

# -------------------------------
# Language Config
# -------------------------------
languages = {
    "Hindi": "hi",
    "Telugu": "te",
    "Tamil": "ta",
    "French": "fr"
}

input_languages = {
    "Auto Detect": None,
    "English": "en",
    "Hindi": "hi",
    "Japanese": "ja"
}

lang_names = {
    "en": "English",
    "hi": "Hindi",
    "te": "Telugu",
    "ta": "Tamil",
    "fr": "French",
    "ja": "Japanese"
}

# Microsoft Neural Voices (natural, high quality)
edge_voices = {
    "Hindi":   "hi-IN-SwaraNeural",
    "Telugu":  "te-IN-ShrutiNeural",
    "Tamil":   "ta-IN-PallaviNeural",
    "French":  "fr-FR-DeniseNeural"
}

# -------------------------------
# Temp File Cleanup
# -------------------------------
temp_files = []

def cleanup_temp_files():
    for f in temp_files:
        try:
            if os.path.exists(f):
                os.remove(f)
        except Exception:
            pass

atexit.register(cleanup_temp_files)

# -------------------------------
# Noise Reduction
# -------------------------------
def reduce_noise(audio_path):
    data, sr = sf.read(audio_path)

    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    reduced = nr.reduce_noise(y=data, sr=sr)

    temp_fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(temp_fd)

    sf.write(path, reduced, sr)
    temp_files.append(path)

    return path

# -------------------------------
# TTS — edge-tts (Neural, Fast)
# -------------------------------
async def _tts_async(text, voice, path):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(path)

def text_to_speech(text, target_lang):
    temp_fd, output_file = tempfile.mkstemp(suffix=".mp3")
    os.close(temp_fd)

    voice = edge_voices[target_lang]

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(_tts_async(text, voice, output_file))
    loop.close()

    # ✅ Ensure file exists and is not empty
    if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
        return None

    temp_files.append(output_file)
    return output_file

# -------------------------------
# Main Pipeline
# -------------------------------
def speech_to_speech(audio, target_lang, input_lang):

    if audio is None or audio == "":
        return "No speech detected", "", "", None

    try:
        # Step 1: Noise Reduction
        audio = reduce_noise(audio)

        # Step 2: ASR with GPU + built-in VAD (replaces silero_vad)
        lang_code = input_languages[input_lang]

        segments, info = model.transcribe(
            audio,
            language=lang_code,
            beam_size=5,              # best accuracy
            vad_filter=True,          # built-in VAD — no silero needed
            vad_parameters=dict(
                min_silence_duration_ms=300   # trim trailing silence quickly
            )
        )

        text = " ".join([seg.text for seg in segments]).strip()

        if text == "":
            return "No clear speech detected", "", "", None

        detected_lang_code = info.language
        detected_lang = lang_names.get(detected_lang_code, detected_lang_code)

        # Step 3: Translation
        translated = GoogleTranslator(
            source='auto',
            target=languages[target_lang]
        ).translate(text)

        # Step 4: Neural TTS via edge-tts
        output_file = text_to_speech(translated, target_lang)

        return detected_lang, text, translated, output_file

    except Exception as e:
        return "Error", str(e), "", None


# -------------------------------
# Gradio UI
# -------------------------------
interface = gr.Interface(
    fn=speech_to_speech,
    inputs=[
        gr.Audio(type="filepath", label="🎙️ Speak (clear and short)"),
        gr.Dropdown(list(languages.keys()), value="Hindi", label="🌐 Target Language"),
        gr.Dropdown(list(input_languages.keys()), value="Auto Detect", label="🔍 Input Language")
    ],
    outputs=[
        gr.Textbox(label="🔎 Detected Language"),
        gr.Textbox(label="📝 Original Speech"),
        gr.Textbox(label="🌍 Translated Text"),
        gr.Audio(label="🔊 Translated Speech", autoplay=True)
    ],
    title="⚡ Speech to Speech Translation",
    description="GPU-accelerated • Neural voices • Built-in VAD • Speak clearly for best accuracy",
    show_progress=True
)

# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":
    interface.launch()