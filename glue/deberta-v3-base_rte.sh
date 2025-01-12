export num_gpus=8
export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0
export output_dir="./glue/debertav3/rte"
python -m torch.distributed.launch --nproc_per_node=$num_gpus \
run_glue.py \
--model_name_or_path /home/wh/zc/model/deberta_v3_base \
--task_name rte \
--do_train \
--do_eval \
--max_seq_length 320 \
--per_device_train_batch_size 32 \
--learning_rate 1.2e-3 \
--num_train_epochs 50 \
--output_dir $output_dir/model \
--overwrite_output_dir \
--logging_steps 10 \
--logging_dir $output_dir/log \
--fp16 \
--evaluation_strategy steps \
--eval_steps 100 \
--save_strategy steps \
--save_steps 10000 \
--warmup_steps 200 \
--cls_dropout 0.2 \
--lora_r 8 \
--lora_alpha 32 \
--seed 0 \
--weight_decay 0.01 \
--target_modules query_proj,key_proj,value_proj