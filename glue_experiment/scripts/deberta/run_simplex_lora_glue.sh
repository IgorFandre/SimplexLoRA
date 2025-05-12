clear
for task_name in rte # cola mnli mrpc qnli qqp rte sst2 stsb
do
    case $task_name in
        mrpc)   num_train_epochs=30
                batch_size=64 ;;
        mnli)   num_train_epochs=7
                batch_size=32 ;;
        qnli)   num_train_epochs=5
                batch_size=32 ;;
        cola)   num_train_epochs=15
                batch_size=64 ;;
        qqp)    num_train_epochs=5
                batch_size=32 ;;
        rte)    num_train_epochs=50
                batch_size=32 ;;
        sst2)   num_train_epochs=10
                batch_size=32 ;;
    esac

    for lr in 5e-5
    # for lr in 3e-5 5e-5 8e-5 1e-4 3e-4 5e-4 8e-4 1e-3
    do
        for seed in 52 1917
        do
            # echo $task_name $lr $num_train_epochs $batch_size
            CUDA_VISIBLE_DEVICES=5 python ./glue_experiment/run_glue.py \
                --dataset_name glue \
                --task_name $task_name \
                --model_name_or_path microsoft/deberta-v3-base \
                --per_device_train_batch_size $batch_size \
                --per_device_eval_batch_size 16 \
                --gradient_accumulation_steps 6 \
                --learning_rate $lr \
                --warmup_steps 30 \
                --learning_rate_w 5e0 \
                --lr_scheduler_type linear \
                --num_train_epochs $num_train_epochs \
                --eval_strategy epoch \
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
                --report_to wandb # none or wandb
        done
    done
done