# AudioMuse-AI-DCLAP

AudioMuse-AI Distilled CLAP (DCLAP) distill the [LAION CLAP](https://github.com/LAION-AI/CLAP) audio tower into a tiny 7 M‑parameter model while retaining its 512‑dimensional embedding space.  The result is a **very fast text‑to‑music search engine** that can run even on a Raspberry PI 5 8GB ram and SSD.

> only the audio branch is distilled; the original text tower is reused (exported from LAION CLAP in `.onnx`).

## Distillation

The goal is simple: distill LAION CLAP's behaviour while cutting the audio tower from ~80 M params to ~7 M and **running 5–6× faster** (performance test done on a Raspberry PI 5 8GB ram and SSD).  A pre‑trained CLAP (`music_audioset_epoch_15_esc_90.14.pt`) supervises a tiny student, initialised with *mn10as* weights from [EfficientAT](https://github.com/fschmid56/EfficientAT).  When this student plateaus on validation cosine, we freeze it and add a second, even smaller student seeded with [EdgeNeXt](https://github.com/mmaaz60/EdgeNeXt) weights.  The newcomer is trained to push the cosine higher until no more gain shows.

The two students are finally fused via a gate that outputs a 512‑dimensional weight vector, deciding—for each feature—how much to trust the second model versus the frozen first.

The dataset for the distillation proces consists of various Creative Commons and public‑domain songs; see the [dataset_license](dataset_license) directory for full details.

The code used for training is in [student_clap](./student_clap) folder.

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

> The text model is exported in .onnx to have all on the same technology but no change was made. The tokenizer is directly downloaded in the code with `AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")` that is a roberta-base tokenizer

Below a code example run on CPU:


```python
import numpy as np
import librosa
import onnxruntime as ort
from transformers import AutoTokenizer

# === CONFIGURATION ===
AUDIO_PATH = "your_audio.wav"          # Path to your audio file
TEXT_QUERY = "Calm piano song"         # Your playlist/mood query string
STUDENT_AUDIO_MODEL = "model_epoch_36.onnx"    # Student audio encoder
TEXT_MODEL = "clap_text_model.onnx"            # Paired ONNX text encoder
SR = 48000
SEGMENT_LENGTH = 480000   # 10 seconds at 48 kHz
HOP_LENGTH = 240000       # 50% overlap
N_MELS = 128
N_FFT = 2048
HOP_LENGTH_MELS = 480

# -- Use the SAME tokenizer as used at student training time --
tokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")  # roberta-base tokenizer

def extract_segmented_logmels(audio_path: str):
    y, _ = librosa.load(audio_path, sr=SR, mono=True)

    # Quantize to int16 and back (matching PyTorch CLAP preprocessing)
    y = np.clip(y, -1.0, 1.0)
    y = (y * 32767.0).astype(np.int16)
    y = (y / 32767.0).astype(np.float32)

    total = len(y)
    segments = []

    if total <= SEGMENT_LENGTH:
        # Short audio: zero-pad to exactly 10 s
        segments.append(np.pad(y, (0, SEGMENT_LENGTH - total)))
    else:
        for start in range(0, total - SEGMENT_LENGTH + 1, HOP_LENGTH):
            segments.append(y[start:start + SEGMENT_LENGTH])
        # Capture the tail segment so the end of the track is never dropped
        last_start = len(segments) * HOP_LENGTH
        if last_start < total:
            segments.append(y[-SEGMENT_LENGTH:])

    mel_segments = []
    for segment in segments:
        mel = librosa.feature.melspectrogram(
            y=segment, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH_MELS,
            win_length=N_FFT, window='hann', center=True, pad_mode='reflect',
            power=2.0, n_mels=N_MELS, fmin=0, fmax=14000,
        )
        log_mel = librosa.power_to_db(mel, ref=1.0, amin=1e-10, top_db=None)
        mel_segments.append(log_mel[np.newaxis, np.newaxis, :, :].astype(np.float32))  # (1, 1, n_mels, time)

    return mel_segments

def cosine_similarity(v1, v2):
    v1 = v1.reshape(-1)
    v2 = v2.reshape(-1)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# -- Load Student ONNX Models --
audio_model = ort.InferenceSession(STUDENT_AUDIO_MODEL)
text_model = ort.InferenceSession(TEXT_MODEL)

# -- AUDIO INFERENCE: feed one segment at a time (model has fixed batch=1) --
mel_segments = extract_segmented_logmels(AUDIO_PATH)
audio_embs = []
for mel_spec in mel_segments:
    emb = audio_model.run(None, {"mel_spectrogram": mel_spec})[0]  # (1, embed_dim)
    audio_embs.append(emb[0])
avg_audio_emb = np.mean(audio_embs, axis=0)                        # (embed_dim,)
avg_audio_emb = avg_audio_emb / (np.linalg.norm(avg_audio_emb) + 1e-9)  # L2 normalize

# -- TEXT ENCODER INFERENCE --
# CLAP encoders require exactly length=77
tok = tokenizer(TEXT_QUERY, padding="max_length", truncation=True, max_length=77, return_tensors="np")
text_emb = text_model.run(None, {
    "input_ids": tok["input_ids"].astype(np.int64),
    "attention_mask": tok["attention_mask"].astype(np.int64),
})[0]  # (1, embed_dim)

# -- MATCH AUDIO & TEXT --
sim = cosine_similarity(avg_audio_emb, text_emb).item()
print(f"Text Query: {TEXT_QUERY}")
print(f"Audio segments processed: {len(mel_segments)}")
print(f"Similarity score: {sim:.4f}")
```

## Acknowledgements

This project was inspired by the [tinyCLAP](https://github.com/fpaissan/tinyCLAP) distillation approach for audio. Their work demonstrated how to compress the audio tower of Microsoft CLAP, and provided the conceptual foundation for DCLAP.
