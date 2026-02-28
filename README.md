# AudioMuse-AI-DCLAP

AudioMuse-AI Distilled CLAP (DCLAP) distill the [**LAION CLAP**](https://github.com/LAION-AI/CLAP) audio tower into a tiny 7 M‑parameter model while retaining its 512‑dimensional embedding space.  The result is a **very fast text‑to‑music search engine**: you type words and retrieve matching music snippets in real time.

> only the audio branch is distilled; the original text tower is reused (exported from LAION CLAP in `.onnx`).

## Distillation

The goal is simple: distill LAION CLAP's behaviour while cutting the audio tower from ~80 M params to ~7 M and running 2–3× faster.  A pre‑trained CLAP (music_audioset_epoch_15_esc_90.14.pt) supervises a tiny student, initialised with *mn10as* weights from [EfficientAT](https://github.com/fschmid56/EfficientAT).  When this student plateaus on validation cosine, we freeze it and add a second, even smaller student seeded with [EdgeNeXt](https://github.com/mmaaz60/EdgeNeXt) weights.  The newcomer is trained to push the cosine higher until no more gain shows.

The two students are finally fused via a gate that outputs a 512‑dimensional weight vector, deciding—for each feature—how much to trust the second model versus the frozen first.

The dataset for the distillation proces consists of various Creative Commons and public‑domain songs; see the [**dataset_license**](dataset_license) directory for full details.

## Quick Start
This command are tested on Linux and MacOS, to be executed on windows they will require some adaptation.

First create a python envirorment and install the dependencies:
```bash
python3 -m venv venv && source venv/bin/activate 
pip install onnxruntime librosa numpy transformers
```

Also get the required data from the release of this project:

| File | Description |
|------|-------------|
| `model_epoch_36.onnx` | Student audio encoder |
| `model_epoch_36.onnx.data` | External weights — **must be in same folder** |
| `clap_text_model.onnx` | Teacher text encoder, from LAION CLAP exported in .onnx |

Using this model the output expected is a 512-dim L2-normalized embeddings on both text and songs audio on which you can run the similarity with dot product. 

> The songs must be splitted in 10s segment with 50% overlap. Each segment is inferenced singularly and then all the segment avaraged together before the similarity check against the text.

Below a code example run on CPU:


```python
import numpy as np
import librosa
import onnxruntime as ort
from transformers import AutoTokenizer

# === CONFIGURATION ===
AUDIO_PATH = "your_audio.wav"       # Path to your audio file
TEXT_QUERY = "Calm piano song"      # Your playlist/mood query string
STUDENT_AUDIO_MODEL = "student_clap_epoch_XX.onnx"  # Your trained ONNX
TEXT_MODEL = "student_clap_text_encoder.onnx"       # Paired ONNX text encoder
SR = 44100
SEGMENT_DURATION = 10.0   # seconds per window
OVERLAP_RATIO = 0.5       # e.g., 0.5 for 50% overlap
N_MELS = 64
N_FFT = 1024
HOP_LENGTH_MELS = 480

# -- Use the SAME tokenizer as used at student training time --
tokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")  # See your config

def extract_segmented_logmels(audio_path: str) -> np.ndarray:
    y, _ = librosa.load(audio_path, sr=SR)
    seg_len = int(SEGMENT_DURATION * SR)
    hop_len = int(seg_len * (1 - OVERLAP_RATIO))
    segments = []
    for i in range(0, len(y) - seg_len + 1, hop_len):
        segment = y[i:i+seg_len]
        mel = librosa.feature.melspectrogram(y=segment, sr=SR, n_fft=N_FFT,
                                             hop_length=HOP_LENGTH_MELS, n_mels=N_MELS)
        log_mel = librosa.power_to_db(mel, ref=np.max)
        # --- CLAP normalization trick, keep in-distribution with teacher ---
        log_mel = ((log_mel + 42.6) / 25.9).T.astype(np.float32)
        segments.append(log_mel[np.newaxis, np.newaxis, :, :]) # [1, 1, Time, Mels]
    return np.concatenate(segments, axis=0) if segments else None

def cosine_similarity(v1, v2):
    v1 = v1.reshape(-1)
    v2 = v2.reshape(-1)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# -- Load Student ONNX Models --
audio_model = ort.InferenceSession(STUDENT_AUDIO_MODEL)
text_model = ort.InferenceSession(TEXT_MODEL)

# -- AUDIO INFERENCE --
audio_segments = extract_segmented_logmels(AUDIO_PATH)
if audio_segments is None:
    raise RuntimeError("Audio too short for configured window length!")
audio_embs = audio_model.run(None, {"input": audio_segments})[0]  # [num_segs, embed_dim]
avg_audio_emb = np.mean(audio_embs, axis=0, keepdims=True)        # [1, embed_dim]

# -- TEXT ENCODER INFERENCE --
# CLAP encoders require exactly length=77s
tok = tokenizer(TEXT_QUERY, padding="max_length", truncation=True, max_length=77, return_tensors="np")
text_emb = text_model.run(None, {
    "input_ids": tok["input_ids"].astype(np.int64),
    "attention_mask": tok["attention_mask"].astype(np.int64),
})[0]  # [1, embed_dim]

# -- MATCH AUDIO & TEXT --
sim = cosine_similarity(avg_audio_emb, text_emb).item()
print(f"Text Query: {TEXT_QUERY}")
print(f"Audio segments processed: {len(audio_embs)}")
print(f"Similarity score: {sim:.4f}")
```

## Acknowledgements

This project was inspired by the [tinyCLAP](https://github.com/fpaissan/tinyCLAP) distillation approach for audio. Their work demonstrated how to compress the audio tower of Microsoft CLAP, and provided the conceptual foundation for DCLAP.