## Description
Here you can see scripts which run tuners with different parameters.

Default run scenario:
```
./run_simplex_lora_glue.sh
```

Obviously, but the name of the script matches the name of the tuner it uses.

Remove cycles for simple experiments.

## SimplexLoRA arguments

* --lora_r:

    Default lora rank.

* --fat_step:

    Number of steps before reranking.

* --max_fat_steps:
    
    Number of rerankings.

* --learning_rate:

    learning rate for all params.

* --learning_rate_w 5e0:
    
    Learning rate for lora weights.

#### Don't change these params if you don't know what they do, use other scripts instead:

* --ft_strategy:

    Name of the tuner. Should be WeightLoRA if you want run SimplexLoRA.

* --lora_extention:

    Type of weight matrix update. Only smart update implemented.

* --use_fat:

    Flag that tells whether to change ranks at all. In SimplexLoRA, ranks always must be changed.

These are the main arguments that affect lora implementation, for other params explanation check /src/config.
