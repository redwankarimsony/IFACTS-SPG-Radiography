#!/bin/bash

models=(
    efficientnet_b1
    efficientnet_b2
    efficientnet_b3
    efficientnet_b4
    efficientnet_b5
    efficientnet_b6
    efficientnet_b7
)

box_presets=(
    t1-t5
    clavicle-only
    complete-vertebrae
)

counter=0
n_processes=7
wait_time=60
experiment_name=base_experiment

# Generate the commands and pipe them to xargs
for model in "${models[@]}"; do
    for preset in "${box_presets[@]}"; do
        if [ $counter -lt $n_processes ]; then
            sleep_duration=$((counter * $wait_time))
            echo "sleep $sleep_duration && python train_base.py --model_arch $model --box_preset $preset --experiment_name $experiment_name"
        else
            echo "python train_base.py --model_arch $model --box_preset $preset --experiment_name $experiment_name"
        fi
        ((counter++))
    done
done | xargs -I CMD -P $n_processes bash -c CMD
