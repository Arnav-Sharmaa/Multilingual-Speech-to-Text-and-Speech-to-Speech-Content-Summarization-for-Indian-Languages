!pip install git+https://github.com/huggingface/parler-tts.git langdetect

!pip install noisereduce

import os
import torch
import torchaudio
import numpy as np
import textwrap
import soundfile as sf
from langdetect import detect
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    MT5Tokenizer,
    MT5ForConditionalGeneration,
    AutoTokenizer
)
from parler_tts import ParlerTTSForConditionalGeneration
import noisereduce as nr
import torchaudio.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

asr_model_dir = "/kaggle/input/whisper-finetuned-model/pytorch/default/1"

summ_model_dir = "/kaggle/input/mt5_xlsum_multilang_model/pytorch/default/1"

asr_processor = WhisperProcessor.from_pretrained("/kaggle/input/whisper-processor/other/default/1")

asr_model = WhisperForConditionalGeneration.from_pretrained(asr_model_dir).to(device).eval()

summ_tokenizer = MT5Tokenizer.from_pretrained(summ_model_dir)

summ_model = MT5ForConditionalGeneration.from_pretrained(summ_model_dir).to(device).eval()

tts_model = ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts").to(device)

desc_tokenizer = AutoTokenizer.from_pretrained(tts_model.config.text_encoder._name_or_path)

prompt_tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")

speaker_map = {
    "hi": ("Rohit", "Hindi"),
    "bn": ("Arjun", "Bengali"),
    "en": ("Thoma", "English"),
    "pa": ("Divjot", "Punjabi"),
    "ta": ("Jaya", "Tamil"),
    "te": ("Prakash", "Telugu"),
    "mr": ("Sanjay", "Marathi")
}

def preprocess_audio(audio_path, target_sr=16000):
    waveform, sr = torchaudio.load(audio_path)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(waveform)

    waveform = waveform.squeeze()

    reduced = nr.reduce_noise(y=waveform.numpy(), sr=target_sr)
    waveform = torch.tensor(reduced)

    waveform = F.vad(waveform, sample_rate=target_sr)
    return waveform

def transcribe(waveform, chunk_duration=30, overlap=3):
    if waveform.dim() > 1:
        waveform = waveform.squeeze()

    chunk_size = chunk_duration * 16000
    step_size = chunk_size - (overlap * 16000)
    total_samples = waveform.shape[-1]

    transcriptions = []
    for start in range(0, total_samples, step_size):
        end = min(start + chunk_size, total_samples)
        chunk = waveform[start:end]

        input_features = asr_processor.feature_extractor(
            chunk.numpy(), sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device)

        with torch.no_grad():
            predicted_ids = asr_model.generate(input_features)
        text = asr_processor.tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
        transcriptions.append(text)

    return " ".join(transcriptions)

def transcribe(audio_path, chunk_duration=30, overlap=3):
    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)
    waveform = waveform.squeeze()

    chunk_size = chunk_duration * 16000
    step_size = chunk_size - (overlap * 16000)
    total_samples = waveform.shape[-1]

    transcriptions = []
    for start in range(0, total_samples, step_size):
        end = min(start + chunk_size, total_samples)
        chunk = waveform[start:end]

        input_features = asr_processor.feature_extractor(
            chunk.numpy(), sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device)

        with torch.no_grad():
            predicted_ids = asr_model.generate(input_features)
        text = asr_processor.tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
        transcriptions.append(text)

    return " ".join(transcriptions)

def summarize(text, ratio=0.3, max_input_length=512):
    inputs = summ_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length,
        padding="max_length"
    ).to(summ_model.device)

    input_length = torch.sum(inputs["attention_mask"]).item()
    dynamic_max_length = min(
        int(input_length * ratio * 1.5),
        256
    )

    generation_config = {
        "max_new_tokens": dynamic_max_length,
        "min_new_tokens": int(dynamic_max_length * 0.7),
        "length_penalty": 0.8 if ratio > 0.5 else 1.2,
        "no_repeat_ngram_size": 3,
        "early_stopping": False
    }

    if ratio > 0.5:
        generation_config.update({
            "do_sample": True,
            "temperature": 0.7 + (ratio-0.5)*0.5,
            "top_k": 40,
            "top_p": 0.9
        })
    else:
        generation_config.update({
            "num_beams": 4,
            "temperature": 0.3
        })

    summary_ids = summ_model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        **generation_config
    )

    decoded = summ_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    last_punct = max(decoded.rfind("."), decoded.rfind("?"), decoded.rfind("!"))
    if last_punct != -1:
        decoded = decoded[:last_punct+1]

    return decoded

def get_description(speaker, lang):
    return f"{speaker}'s voice in {lang} ."

def synthesize(text, speaker, lang, max_length=512):
    description = get_description(speaker, lang)
    desc_ids = desc_tokenizer(description, return_tensors="pt").to(device)

    chunks = textwrap.wrap(text, width=max_length, break_long_words=False)
    chunk_audios = []

    for chunk in chunks:
        prompt_ids = prompt_tokenizer(chunk, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        audio = tts_model.generate(
            input_ids=desc_ids.input_ids,
            attention_mask=desc_ids.attention_mask,
            prompt_input_ids=prompt_ids.input_ids,
            prompt_attention_mask=prompt_ids.attention_mask
        ).cpu().numpy().squeeze()
        chunk_audios.append(audio)

    return np.concatenate(chunk_audios, axis=-1)

def process_audio_to_summary_tts(audio_file, output_folder="output"):
    os.makedirs(output_folder, exist_ok=True)
    print("\nStep 0: Preprocessing audio.")
    waveform = preprocess_audio(audio_file)
    print("\nStep 1: Transcribing audio.")
    transcription = transcribe(audio_file)
    print("\n--- Transcription ---\n", transcription)
    print()
    print("\nStep 2: Summarizing transcription.")
    summary = summarize(transcription)
    print("\n--- Summary ---\n", summary)

    print("\nStep 3: Detecting language of transcription.")
    lang_code = detect(transcription)
    speaker, lang = speaker_map.get(lang_code, ("Anita", "Hindi"))
    print(f"Selected voice: {speaker} ({lang})")

    print("\nStep 4: Synthesizing audio for transcription.")
    input_audio = synthesize(transcription, speaker, lang)
    input_path = os.path.join(output_folder, "input.wav")
    sf.write(input_path, input_audio, tts_model.config.sampling_rate)

    print("\nStep 5: Synthesizing audio for summary.")
    summary_audio = synthesize(summary, speaker, lang, max_length=256)
    summary_path = os.path.join(output_folder, "summary.wav")
    sf.write(summary_path, summary_audio, tts_model.config.sampling_rate)

    with open(os.path.join(output_folder, "transcription.txt"), "w", encoding="utf-8") as f:
        f.write(transcription)

    with open(os.path.join(output_folder, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary)

    print(f"\n Process completed. Output files saved in: {output_folder}")

    print("\nPlaying synthesized transcription audio:")
    display(ipd.Audio(input_path))

    print("\n Playing synthesized summary audio:")
    display(ipd.Audio(summary_path))

audio_file = "/kaggle/input/speech-summarization-test-dataset/sample_0004/input.wav"

process_audio_to_summary_tts(audio_file)

