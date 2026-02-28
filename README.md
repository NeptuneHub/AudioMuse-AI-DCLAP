# AudioMuse-AI-DCLAP

AudioMuse-AI Distilled CLAP (DCLAP) distill the **LAION CLAP** audio tower into a tiny 7 M‑parameter model while retaining its 512‑dimensional embedding space.  The result is a **very fast text‑to‑music search engine**: you type words and retrieve matching music snippets in real time.

> only the audio branch is distilled; the original text tower is reused (exported from LAION CLAP in `.onnx`).

Source CLAP repo: https://github.com/LAION-AI/CLAP

## Distillation

The goal is simple: distill LAION CLAP's behaviour while cutting the audio tower from ~80 M params to ~7 M and running 2–3× faster.  A pre‑trained CLAP (music_audioset_epoch_15_esc_90.14.pt) supervises a tiny student, initialised with *mn10as* weights from EfficientAT (https://github.com/fschmid56/EfficientAT).  When this student plateaus on validation cosine, we freeze it and add a second, even smaller student seeded with **EdgeNeXt** weights (https://github.com/mmaaz60/EdgeNeXt).  The newcomer is trained to push the cosine higher until no more gain shows.

The two students are finally fused via a gate that outputs a 512‑dimensional weight vector, deciding—for each feature—how much to trust the second model versus the frozen first.

The dataset for the distillation proces consists of various Creative Commons and public‑domain songs; see the [**dataset_license**](dataset_license) directory for full details.

## Acknowledgements

This project was inspired by the **tinyCLAP** distillation approach for audio:

- https://github.com/fpaissan/tinyCLAP

Their work demonstrated how to compress the audio tower of Microsoft CLAP, and provided the conceptual foundation for DCLAP.