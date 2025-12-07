# inference_dreambooth.py --model-id outputs/dreambooth/dog --prompt "A photo of sks dog in a bucket" --steps 50 --guidance 7.5 --seed 42 --out dog-bucket.png
import argparse
from diffusers import StableDiffusionPipeline
import torch
from pytorch_lightning import seed_everything

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="outputs/dreambooth/dog")
    p.add_argument("--prompt", default="A photo of sks dog in a bucket")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="dog-bucket.png")
    args = p.parse_args()

    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_id, torch_dtype=torch.float16
    ).to("cuda")

    seed_everything(args.seed)
    img = pipe(args.prompt, num_inference_steps=args.steps, guidance_scale=args.guidance).images[0]
    img.save(args.out)
    print(f"saved: {args.out}")
