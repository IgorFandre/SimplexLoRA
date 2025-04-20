clear
for task_name in mrpc
do
    CUDA_VISIBLE_DEVICES=6 python ./glue_experiment/run_glue.py\
        --dataset_name glue \
        --task_name $task_name \
        --model_name_or_path microsoft/deberta-v3-base \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 16 \
        --gradient_accumulation_steps 6 \
        --learning_rate 8e-4 \
        --lr_scheduler_type linear \
        --warmup_steps 100 \
        --max_steps 512 \
        --eval_steps 64 \
        --save_steps 256 \
        --ft_strategy DoRA \
        --lora_r 8 \
        --lora_alpha 32 \
        --lora_dropout 0.05 \
        --report_to wandb # none or wandb
done