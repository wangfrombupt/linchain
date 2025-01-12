export num_gpus=8
export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0
export output_dir="./glue_result/debertav3/qqp"
python -m torch.distributed.launch --nproc_per_node=$num_gpus --master_port 29509 \
run_glue.py \
--model_name_or_path deberta_v3_base \
--task_name qqp \
--do_train \
--do_eval \
--max_seq_length 320 \
--per_device_train_batch_size 8 \
--learning_rate 4e-4 \
--num_train_epochs 20 \
--output_dir $output_dir/model \
--logging_steps 500 \
--logging_dir $output_dir/log \
--fp16 \
--evaluation_strategy steps \
--eval_steps 3000 \
--save_strategy steps \
--save_steps 10000 \
--warmup_steps 2000 \
--cls_dropout 0.2 \
--lora_r 8 \
--lora_alpha 32 \
--seed 0 \
--weight_decay 0.01 \

# --overwrite_output_dir \