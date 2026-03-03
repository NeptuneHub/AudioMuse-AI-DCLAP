# Student CLAP: Lightweight Audio Encoder

This is the code used for the distillation process of the Audio Tower of LAION CLAP.

## Quick Start

With this command you will create the virtual env with all the dependencies and start the training:

```bash
# Setup and install
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt

# Run training
screen -S training
python3 train_real.py --config config.yaml
screen -r training

```

To run report you can use this (you need to change the name of the .onnx model in the code)
```
python final_test.py
```

To re run only the validation on a checkpoint
```
python3 revalidate_checkpoint.py --ckpt checkpoint_epoch_1.pth
```

## Useful command

You can check how the average cosine similarity (training and validation) is going for each epoch with this one line command:
```
python3 - <<'PY'
import glob, torch
for f in sorted(glob.glob('student_clap/checkpoints/checkpoint_epoch_*.pth')):
    ckpt = torch.load(f, map_location='cpu')
    m = ckpt.get('train_metrics', {})
    val_mse = ckpt.get('val_mse', ckpt.get('last_val_mse', ckpt.get('best_val_mse','N/A')))
    val_cos = ckpt.get('val_cosine_sim', ckpt.get('last_val_cosine', ckpt.get('best_val_cosine','N/A')))
    print(f"{f}: train_cos={m.get('avg_cosine_sim')}, train_mse={m.get('avg_mse')}, val_mse={val_mse}, val_cos={val_cos}, lr={m.get('learning_rate')}")
PY
```

with semantic metrics
```
python3 - <<'PY'
import glob, torch
for f in sorted(glob.glob('student_clap/checkpoints/checkpoint_epoch_*.pth')):
    ckpt = torch.load(f, map_location='cpu')
    m = ckpt.get('train_metrics', {})
    val_mse = ckpt.get('val_mse', ckpt.get('last_val_mse', 'N/A'))
    val_sem = ckpt.get('val_semantic_error', 'N/A')
    print(f"{f}: train_cos={m.get('avg_cosine_sim')}, train_mse={m.get('avg_mse')}, train_sem={m.get('avg_semantic','N/A')}, val_mse={val_mse}, val_sem={val_sem}, lr={m.get('learning_rate')}")
PY
```

with both:
```
python3 - <<'PY'
import glob, torch
for f in sorted(glob.glob('student_clap/checkpoints/checkpoint_epoch_*.pth')):
    ckpt = torch.load(f, map_location='cpu')
    m = ckpt.get('train_metrics', {})
    val_mse = ckpt.get('val_mse', ckpt.get('last_val_mse', 'N/A'))
    val_cos = ckpt.get('val_cosine', 'N/A')
    val_met = ckpt.get('val_metric', 'N/A')
    val_met_name = ckpt.get('val_metric_name', 'N/A')
    val_sem = ckpt.get('val_semantic_error', 'N/A')
    print(f"{f}: train_cos={m.get('avg_cosine_sim')}, train_mse={m.get('avg_mse')}, train_sem={m.get('avg_semantic','N/A')}, val_mse={val_mse}, {val_met_name}={val_met}, val_sem={val_sem}, lr={m.get('learning_rate')}")
PY
```

You can check the million of parameter used for your input configuration with this command:
```
PYTHONPATH=.. python -c "import yaml; from student_clap.models.student_onnx_model import StudentCLAPAudio; config=yaml.safe_load(open('config.yaml')); m=StudentCLAPAudio(config); print(m.count_parameters())"
```

To check instead which configuration of input you used for a checkpoint you can use this command:
```
PYTHONPATH=.. python -c "import torch; m=torch.load('student_clap/checkpoints/CHECKPOINT-NAME-HERE.pth', map_location='cpu'); print({k: v for k, v in m['config']['model'].items() if k.startswith('efficientat_') or k=='efficientat_model'})"
```

To force the algorithm to read LR from config.yaml after a stop, instead of reading from the scheduler:
```
python3 - <<'PY'
import torch, glob
for p in glob.glob('student_clap/checkpoints/checkpoint_epoch_*.pth'):
    ckpt = torch.load(p, map_location='cpu')
    ckpt.pop('optimizer_state_dict', None)
    ckpt.pop('scheduler_state_dict', None)
    torch.save(ckpt, p)
    print("Stripped optimizer/scheduler from", p)
PY
```

To force the algorithm to read the weight decay value from config.yaml after a stop, instead of reading from schedule:
```
python3 - <<'PY'
import torch, glob, shutil
paths = glob.glob('student_clap/checkpoints/checkpoint_epoch_*.pth') + ['student_clap/checkpoints/latest.pth']
for p in paths:
    try:
        shutil.copy(p, p + '.bak')
        ckpt = torch.load(p, map_location='cpu')
        changed = False
        for k in ('scheduler_state_dict','optimizer_state_dict'):
            if ckpt.pop(k, None) is not None:
                changed = True
        if changed:
            torch.save(ckpt, p)
            print('Cleaned', p)
        else:
            print('No optimizer/scheduler state in', p)
    except Exception as e:
        print('Skipped', p, ':', e)
PY
```
Reset logit scale to initial value:
```
python3 -c "
import torch
ckpt = torch.load('student_clap/checkpoints/latest.pth', map_location='cpu')
ckpt['model_state_dict']['logit_scale'] = torch.tensor(2.6592)  # Reset to init
torch.save(ckpt, 'student_clap/checkpoints/latest.pth')
print('Reset logit_scale to 2.6592')
"
```

Clear Adamw optimizer
```
python -c "
import torch
ck = torch.load('student_clap/checkpoints/checkpoint_epoch_9.pth', map_location='cpu')
ck.pop('optimizer_state_dict', None)
ck.pop('scheduler_state_dict', None)
ck['patience_counter'] = 0
torch.save(ck, 'student_clap/checkpoints/checkpoint_epoch_9_nostate.pth')
print('Keys saved:', list(ck.keys()))
"
```

Check the cosine and val cosine also in subfolder:
```
find student_clap/checkpoints -name "checkpoint_epoch_*.pth" | sort -V | python3 -c '
import torch, sys
for line in sys.stdin:
    f = line.strip()
    try:
        ckpt = torch.load(f, map_location="cpu", weights_only=False)
        m = ckpt.get("train_metrics", {})
        avg = m.get("avg_cosine_sim", "null")
        lr = m.get("learning_rate", "null")
        val = ckpt.get("last_val_cosine", ckpt.get("val_cosine_sim", ckpt.get("best_val_cosine", "null")))
        print(f"{f}: cosine={avg}, val_cosine={val}, lr={lr}")
    except Exception as e:
        print(f"{f}: ERROR - {e}")
'
```

## Training

### Songs

**Architecture**
- Base student: EfficientAT MobileNet (default `mn10_as`, n_mels=128 -> 512‑dim).  Optional pretrained variants are selectable via `model.efficientat_model`.
- *Fusion mode* (enabled when `model.specialist_checkpoint` is non‑null): a **frozen specialist** (previously trained EfficientAT) is paired with a lightweight trainable student backbone.  The current default backbone is EdgeNeXt‑XX‑Small (`fusion_backbone: edgenext`), but `efficientat`, `deit_tiny` and `mobilevitv2` are also supported.  A 512‑dim per‑channel gate blends specialist and student outputs.
- Projection head: residual MLP (backbone_dim→512), dropout and bias configurable.  Final embeddings are L2‑normalized.

**Training schedule**
1. **Stage 1** – distill CLAP teacher across all parameters; length controlled by `training.epochs`.
2. **Stage 2** – triggered automatically after stage 1; encoder weights are frozen and only the projection/gate (in fusion) remain trainable.  Duration, learning rate and scheduler are configured via `training.stage2_*` fields.  The shortcut flag `training.projection_only` can force stage 2 behaviour on startup.

Warmup, LR scheduling (ReduceLROnPlateau by default or optional CosineAnnealingLR), mixed precision, augmentation, mixup and other hyper‑parameters live entirely in `config.yaml`.

**Segmentation & batching**
10‑second segments with 50 % overlap; segments are processed in small groups (`model.segment_batch_size`) and aggregated either individually, averaged or both (`training.training_strategy`).

**Loss & scaling options**
- Loss function selectable between `cosine`, `mse` or `kl` (cosine/KL disable semantic loss).
- Static temperature (`loss_temperature`) or learnable logit-scale (`use_logit_scale` + `init_logit_scale`, clamped by `max_logit_scale_T`).
- Optional focal weighting, embedding normalization, semantic alignment (`lambda_semantic`), and teacher cache toggles.