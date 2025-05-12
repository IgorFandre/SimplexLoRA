clear
for task_name in rte # cola mnli mrpc qnli qqp rte sst2 stsb
do
    case $task_name in
        mrpc)   batch_size=64 ;;
        mnli)   batch_size=32 ;;
        qnli)   batch_size=32 ;;
        cola)   batch_size=64 ;;
        qqp)    batch_size=32 ;;
        rte)    batch_size=8 ;;
        sst2)   batch_size=32 ;;
    esac

    for lr in 1e-5 #3e-5 5e-5 8e-5 1e-4
    # for lr in 3e-5 5e-5 8e-5 1e-4 3e-4 5e-4 8e-4 1e-3
    do
        for seed in 18
        do
            # echo $task_name $lr $num_train_epochs $batch_size
            export HF_TOKEN=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
            CUDA_VISIBLE_DEVICES=1 python ./glue_experiment/run_glue.py \
                --dataset_name glue \
                --task_name $task_name \
                --model_name_or_path meta-llama/Meta-Llama-3.1-8B \
                --per_device_train_batch_size $batch_size \
                --per_device_eval_batch_size 16 \
                --gradient_accumulation_steps 16 \
                --learning_rate $lr \
                --warmup_steps 30 \
                --learning_rate_w 5e0 \
                --lr_scheduler_type cosine \
                --max_seq_length 1024 \
                --max_steps 256 \
                --eval_strategy steps \
                --eval_step 64 \
                --max_val_samples 101 \
                --save_strategy no \
                --ft_strategy WeightLoRA \
                --lora_r 8 \
                --lora_dropout 0.05 \
                --lora_alpha 32 \
                --use_fat true \
                --fat_step 5 \
                --max_fat_steps 3 \
                --lora_extention smart \
                --seed $seed \
                --do_eval true \
                --do_predict false \
                --bf16 true \
                --report_to none # none or wandb
        done
    done
done