# AudioMuse-AI-DCLAP

AudioMuse-AI Distilled CLAP (DCLAP) is a standalone effort to distill the **LAION CLAP** model, with a focus exclusively on **music** audio embeddings.

- Source repository: https://github.com/LAION-AI/CLAP

The distillation process aims to retain accuracy comparable to the original network while dramatically reducing its footprint: the audio tower shrinks from roughly 80 million parameters down to about 7 million, and inference becomes 2×–3× faster depending on hardware.

The workflow works in stages. a pre‑trained CLAP network (music_audioset_epoch_15_esc_90.14.pt) serves as the teacher.  We then train a small “student” network starting from the *mn10as* weights from EfficientAT:

- EfficientAT base: https://github.com/fschmid56/EfficientAT

The training is iterative: once the student stops improving its **validation cosine** score (i.e. it has reached its capacity), we freeze its parameters and stack a second, even smaller student model on top.  This secondary student is initialised with the **EdgeNeXt** weights:

- EdgeNeXt base: https://github.com/mmaaz60/EdgeNeXt

It is then trained to push the validation cosine as high as possible, yielding the final distilled model.

The idea is therefore to build the full DCLAP model by sequentially adding and freezing students, each one contributing additional capacity until no further gain is possible.

In the final architecture the two student models are merged through a **fusion layer**: a gated mechanism produces a 512‑dimensional weight vector that determines, for each feature dimension, how much to rely on the second (EdgeNeXt‑based) student versus the frozen first student.

> **Note:** only the **audio tower** of the original LAION CLAP is distilled.  The text tower remains unchanged and is used directly from the original LAION CLAP model, simply exported to `.onnx` format for compatibility.

***

### Acknowledgements

This project was inspired by the **tinyCLAP** distillation approach for audio:

- https://github.com/fpaissan/tinyCLAP

Their work demonstrated how to compress the audio tower of Microsoft CLAP, and provided the conceptual foundation for DCLAP.

The dataset used for the distillation consists of various Creative Commons and public‑domain songs; see the [**dataset_license**](dataset_license) directory for full details.