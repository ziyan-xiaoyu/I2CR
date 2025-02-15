export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
CUDA_VISIBLE_DEVICES=1 llamafactory-cli train \
    --stage sft \
    --do_train \
    --model_name_or_path /vicuna-7b-v1.5 \
    --dataset wikidiverse_train_v6_summary_nil \
    --dataset_dir /LLaMA-Factory/data \
    --template vicuna \
    --finetuning_type lora \
    --output_dir /model_arg/vicuna15_7B/checkpoint_1 \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 4096 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --warmup_steps 100 \
    --save_steps 500 \
    --eval_steps 500 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --learning_rate 2e-5 \
    --num_train_epochs 2.0 \
    --val_size 0.1 \
    --plot_loss \
    --fp16 \
    # --ddp_timeout 180000000
