# Multilingual Speech-to-Text and Speech-to-Speech Content Summarization for Indian Languages

## Project Overview
This project develops a multilingual pipeline for speech-to-text (STT) and speech-to-speech (STS) summarization tailored for Indian languages. The pipeline leverages state-of-the-art models to transcribe audio, summarize content, and synthesize summaries back into speech. Key components include:

- **Speech-to-Text**: Fine-tuned Whisper ASR model for accurate transcription of audio in multiple Indian languages.
- **Text Summarization**: mT5 model for generating concise summaries of transcribed text.
- **Text-to-Speech**: Indic Parler-TTS for synthesizing summaries into natural-sounding speech.

Supported languages include English, Hindi, Punjabi, Urdu, Bengali, Tamil, Telugu, and Marathi.

## Repository Structure
- `asr.ipynb`: Notebook for fine-tuning the Whisper model for automatic speech recognition (ASR).
- `summarization.ipynb`: Notebook for training the mT5 model for text summarization.
- `dataset_generation.py`: Script to generate a dataset with audio inputs and summaries using Indic Parler-TTS.
- `pipeline.py`: Main script integrating ASR, summarization, and TTS for end-to-end processing.

## Installation
To set up the project, install the required dependencies:

```bash
pip install transformers datasets jiwer torchaudio soundfile matplotlib sacrebleu rouge_score torch
pip install git+https://github.com/huggingface/parler-tts.git
pip install noisereduce langdetect
```

Ensure you have a CUDA-enabled GPU for optimal performance, as the models leverage GPU acceleration where available.

## Usage
### 1. Automatic Speech Recognition (`asr.ipynb`)
This notebook fine-tunes the Whisper-small model on the FLEURS dataset for Indian languages:
- **Input**: Audio files in supported languages.
- **Process**: Preprocesses audio, fine-tunes the Whisper model, and evaluates using Word Error Rate (WER).
- **Output**: Transcriptions and a fine-tuned model saved at `/kaggle/working/whisper-finetuned-model`.

Run the notebook to train and save the model:
```bash
jupyter notebook asr.ipynb
```

### 2. Text Summarization (`summarization.ipynb`)
This notebook trains the mT5 model for summarization using the XLSum dataset:
- **Input**: Text articles in supported languages.
- **Process**: Maps articles to summaries, trains mT5, and evaluates using ROUGE and BLEU scores.
- **Output**: Summarized text and a trained model.

Run the notebook to train the summarization model:
```bash
jupyter notebook summarization.ipynb
```

### 3. Dataset Generation (`dataset_generation.py`)
This script generates a dataset with audio inputs and summaries:
- **Input**: Text articles from the XLSum dataset.
- **Process**: Converts text to audio using Indic Parler-TTS and saves input text, summaries, and audio files.
- **Output**: Organized dataset in `dataset/` with subfolders containing `input.txt`, `summary.txt`, `input.wav`, `summary.wav`, and `lang.txt`.

Execute the script:
```bash
python dataset_generation.py
```

### 4. End-to-End Pipeline (`pipeline.py`)
This script integrates all components for processing audio to summarized speech:
- **Input**: Audio file (e.g., WAV format).
- **Process**:
  1. Preprocesses audio (noise reduction, voice activity detection).
  2. Transcribes audio using the fine-tuned Whisper model.
  3. Summarizes transcription using the mT5 model.
  4. Detects language and selects an appropriate speaker.
  5. Synthesizes transcription and summary into audio using Indic Parler-TTS.
- **Output**: Transcription (`transcription.txt`), summary (`summary.txt`), and audio files (`input.wav`, `summary.wav`) saved in the `output/` folder.

Run the pipeline:
```bash
python pipeline.py
```

Example usage:
```python
audio_file = "/path/to/input.wav"
process_audio_to_summary_tts(audio_file, output_folder="output")
```

## Models
- **Whisper**: Fine-tuned `openai/whisper-small` for ASR, available at `/kaggle/input/whisper-finetuned-model`.
- **mT5**: Trained for summarization, available at `/kaggle/input/mt5_xlsum_multilang_model`.
- **Indic Parler-TTS**: Pre-trained `ai4bharat/indic-parler-tts` for speech synthesis.

## Dataset
- **FLEURS**: Used for ASR training, covering multiple Indian languages.
- **XLSum**: Used for summarization training and dataset generation, providing articles and summaries in supported languages.
- **Generated Dataset**: Created by `dataset_generation.py`, stored in `dataset/` with audio and text files.

## Requirements
- Python 3.8+
- PyTorch with CUDA support (optional for GPU acceleration)
- Kaggle API for accessing pre-trained models (see `pipeline.py` for Kaggle dataset imports)

## Notes
- The pipeline is optimized for Indian languages but can be extended to other languages by updating the model configurations and datasets.
- Ensure sufficient disk space for storing generated audio files and model checkpoints.
- For production use, consider optimizing the chunking strategy in `pipeline.py` for long audio files.

## License
This project is licensed under the MIT License.
