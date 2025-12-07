"""
inference_tuneqdm.py

Run Stable Diffusion with TuneQDM.
Quantize UNet, load trained expert scales, and ensure per-step expert selection via forward pre-hook.
"""

import argparse
from typing import Optional, Tuple

import torch
from diffusers import StableDiffusionPipeline
from tuneqdm import TuneQDMConfig, quantize_model, load_delta_state, set_timestep


def load_config_from_checkpoint(path: str) -> Optional[TuneQDMConfig]:
    """Load config from checkpoint without loading weights."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "config" in checkpoint:
        return TuneQDMConfig(**checkpoint["config"])
    return None


# --------------------------- Utility funcs ---------------------------- #

def set_seed_all(seed: int) -> None:
    """Set torch random seed."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def register_tas_pre_hook(
    unet: torch.nn.Module,
    total_denoising_steps: int
):
    """Register a forward pre-hook to set TAS expert per UNet call.

    The hook reads `timestep` from kwargs if present, otherwise from the 2nd positional arg.
    It then dispatches set_timestep(unet, t, T) so that all TQLinear/TQConv2d layers pick the right expert.

    Args:
        unet: Quantized UNet module.
        total_denoising_steps: T (e.g., 1000).
    """
    def _pre_hook(module: torch.nn.Module, args: Tuple, kwargs: dict) -> None:
        """Forward pre-hook to propagate `timestep` to TAS layers."""
        if "timestep" in kwargs:
            t_value = kwargs["timestep"]
        elif len(args) >= 2:
            t_value = args[1]
        else:
            return
        set_timestep(module, t_value, total_denoising_steps=total_denoising_steps)

    # with_kwargs=True is important because diffusers usually calls with keyword args
    handle = unet.register_forward_pre_hook(_pre_hook, with_kwargs=True)
    return handle


# ------------------------------ Main ---------------------------------- #

def main() -> None:
    """Main entry for TAS inference."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="outputs/tuneqdm/dog")
    parser.add_argument("--prompt", type=str, default="A photo of sks dog in a bucket")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="dog-bucket-tuneqdm.png")
    args = parser.parse_args()

    set_seed_all(args.seed)

    state_path = f"{args.model_id}/tuneqdm_state.pt"

    # 1) Load config from checkpoint
    cfg = load_config_from_checkpoint(state_path)
    if cfg is None:
        raise ValueError(f"No config found in {state_path}. Was it trained with the latest version?")

    # 2) Load pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
        low_cpu_mem_usage=False,
    ).to("cuda")

    # 3) Quantize UNet with loaded config
    quantize_model(pipe.unet, cfg)

    # 4) Load trained expert deltas
    load_delta_state(pipe.unet, state_path)

    # 5) Ensure eval + dtype
    pipe.unet = pipe.unet.to("cuda").to(torch.float16)
    pipe.unet.eval()

    # 6) Register TAS pre-hook
    _hook = register_tas_pre_hook(pipe.unet, total_denoising_steps=cfg.total_denoising_steps)

    # 7) Run generation
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    image = pipe(
        args.prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        generator=generator,
    ).images[0]
    image.save(args.out)
    print(f"saved: {args.out}")


# --------------------------- Example usage ----------------------------- #

if __name__ == "__main__":
    main()
