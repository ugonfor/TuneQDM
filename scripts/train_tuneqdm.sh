instance=$1

export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="dataset/$instance"
export OUTPUT_DIR="outputs/tuneqdm/$instance"

uv run accelerate launch tuneqdm/train_tuneqdm.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks $instance" \
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