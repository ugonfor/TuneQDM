# TuneQDM

Official PyTorch implementation of **"Memory-Efficient Fine-Tuning for Quantized Diffusion Model"** (ECCV 2024)
- Since my codebase was not really clean, I refactored my code with claude, and I have checked the implementation is same with mine.

## Overview

TuneQDM enables memory-efficient fine-tuning of diffusion models through weight-only INT quantization with trainable delta scales. Instead of fine-tuning all model weights, TuneQDM:

1. **Quantizes** the UNet weights to INT (4, 8-bit supported)
2. **Freezes** the quantized weights
3. **Trains only** the scale parameters (delta_out, delta_in)

This approach dramatically reduces memory usage while maintaining generation quality.

### Key Features

- **Weight-only INT Quantization**: Supports 4, 8-bit quantization
- **Trainable Delta Scales**: Only scale parameters are trained, not the quantized weights
- **TAS (Timestep-Aware Scales)**: Multiple expert copies of scales with automatic selection based on diffusion timestep

## Installation

```bash
# Clone repository
git clone https://github.com/your-repo/TuneQDM.git
cd TuneQDM

# Install dependencies with uv
uv sync
```

## Quick Start

### 1. Prepare Dataset

Download sample dataset from Google's DreamBooth repository:

```bash
git clone https://github.com/google/dreambooth.git
mv dreambooth/dataset .
rm -rf dreambooth
```

Dataset structure:
```
dataset/
├── dog/          # 5 images of a specific dog
├── cat/          # 5 images of a specific cat
└── ...
```

### 2. Training

#### TuneQDM (Quantized Fine-tuning)

```bash
# Using the provided script
bash scripts/train_tuneqdm.sh dog

# Or run directly with custom parameters
uv run accelerate launch tuneqdm/train_tuneqdm.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"  \
  --instance_data_dir=dataset/dog \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=3e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=400 \
  --num_experts=4 \
  --n_bits=8 \
  --train_delta_in \
  --seed 42
```

### 3. Inference

#### TuneQDM

```bash
uv run python tuneqdm/inference_tuneqdm.py \
  --model-id outputs/tuneqdm/dog \
  --prompt "A photo of sks dog in a bucket" \
  --steps 50 \
  --guidance 7.5 \
  --seed 42 \
  --out output.png
```

## Configuration

### TuneQDM Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_bits` | 8 | Quantization bit-width (2, 4, 8) |
| `num_experts` | 1 | Number of TAS experts (1 = no TAS) |
| `train_delta_in` | False | Train input channel scales (delta_in) |
| `train_delta_out` | True | Train output channel scales (delta_out) |
| `total_denoising_steps` | 1000 | Total diffusion timesteps for expert selection |

### Expert Selection (TAS)

When `num_experts > 1`, the expert is selected based on the current timestep:

```
expert_idx = floor(t * num_experts / total_denoising_steps)
```

This allows different scales for different noise levels during the diffusion process.

## Architecture

### Core Components

```
tuneqdm/
├── core.py              # TuneQDM core: TQLinear, TQConv2d, quantization
├── train_tuneqdm.py     # Training script with TuneQDM
├── train_dreambooth.py  # Baseline DreamBooth training
├── inference_tuneqdm.py # Inference with TuneQDM models
└── inference_dreambooth.py # Inference with DreamBooth models
```

### Quantized Layers

- **TQLinear**: Quantized Linear layer with expert scales
- **TQConv2d**: Quantized Conv2d layer with expert scales

Each layer stores:
- `weight_int`: INT8 quantized weights (frozen)
- `delta_out`: Output channel scales (trainable)
- `delta_in`: Input channel scales (optional, trainable)
- `experts`: Number of expert copies for TAS

### Checkpoint Format

TuneQDM saves only the trainable parameters with configuration:

```python
{
    "delta_state": {
        "layer1.delta_out": tensor,
        "layer1.delta_in": tensor,
        ...
    },
    "config": {
        "n_bits": 8,
        "num_experts": 4,
        "train_delta_in": True,
        ...
    }
}
```

## API Usage

### Programmatic Training

```python
from tuneqdm.core import TuneQDMConfig, apply_tuneqdm, set_timestep, save_delta_state

# Configure
config = TuneQDMConfig(
    n_bits=8,
    num_experts=4,
    train_delta_in=True
)

# Apply quantization to UNet
apply_tuneqdm(unet, config)

# Training loop
for batch in dataloader:
    set_timestep(unet, timestep)  # Set expert based on timestep
    loss = compute_loss(unet, batch)
    loss.backward()
    optimizer.step()

# Save
save_delta_state(unet, "tuneqdm_state.pt", config=config)
```

### Programmatic Inference

```python
from tuneqdm.core import TuneQDMConfig, quantize_model, load_delta_state, register_tas_pre_hook

# Load config from checkpoint
checkpoint = torch.load("tuneqdm_state.pt")
config = TuneQDMConfig(**checkpoint["config"])

# Apply quantization and load weights
quantize_model(pipe.unet, config)
load_delta_state(pipe.unet, "tuneqdm_state.pt")

# Register automatic timestep dispatch
register_tas_pre_hook(pipe.unet, config.num_experts, num_inference_steps=50)

# Generate
image = pipe(prompt, num_inference_steps=50).images[0]
```

## Citation

```bibtex
@inproceedings{ryu2024tuneqdm,
  title={Memory-efficient fine-tuning for quantized diffusion model},
  author={Ryu, Hyogon and Lim, Seohyun and Shim, Hyunjung},
  booktitle={European Conference on Computer Vision},
  pages={356--372},
  year={2024},
  organization={Springer}
}
```

## License

This project is licensed under the MIT License.
