export num_gpus=8
export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0
export output_dir="./glue_result/debertav3/cola"
python -m torch.distributed.launch --nproc_per_node=$num_gpus  --master_port 29511 \
run_glue.py \
--model_name_or_path deberta_v3_base \
--task_name cola \
--do_train \
--do_eval \
--max_seq_length 64 \
--per_device_train_batch_size 32 \
--learning_rate 1.3e-4 \
--num_train_epochs 25 \
--output_dir $output_dir/model \
--overwrite_output_dir \
--logging_steps 10 \
--logging_dir $output_dir/log \
--fp16 \
--evaluation_strategy steps \
--eval_steps 100 \
--save_strategy steps \
--save_steps 10000 \
--warmup_steps 100 \
--cls_dropout 0.1 \
--lora_r 8 \
--lora_alpha 32 \
--seed 0 \
--weight_decay 0 \
--use_deterministic_algorithms