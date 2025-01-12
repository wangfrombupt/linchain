#!/bin/sh

rank=16
alpha=32
gpuid=0
mixer_num=3

model_p_or_n=xxxx

model_path=trained_models/linchain-$mixer_num-r$rank-a$alpha-3e4
results_path=results/linchain-$mixer_num-r$rank-a$alpha-3e4

mkdir -p $model_path
mkdir -p $results_path


CUDA_VISIBLE_DEVICES=$gpuid python -u finetune.py \
  --base_model $model_p_or_n \
  --data_path 'ft-training_set/math_10k.json' \
  --output_dir $model_path \
  --batch_size 16 \
  --micro_batch_size 4 \
  --num_epochs 3 \
  --learning_rate 3e-4 \
  --cutoff_len 256 \
  --val_set_size 120 \
  --adapter_name lora \
  --lora_r $rank \
  --lora_alpha $alpha \
  --use_linchain \
  --mixer_num $mixer_num \
  --target_modules "["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]"


for ds in AQuA gsm8k SVAMP AddSub MultiArith SingleEq mawps
do
  CUDA_VISIBLE_DEVICES=$gpuid python -u evaluate.py \
    --model LLaMA3 \
    --adapter LoRA \
    --dataset $ds \
    --base_model $model_p_or_n \
    --lora_weights $model_path \
    --save_dir $results_path
done