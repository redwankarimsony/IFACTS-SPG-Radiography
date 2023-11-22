#!/bin/bash

models=(
    densenet121
    densenet161
    densenet169
    densenet201
    efficientnet_b0
    resnet34
    resnet50
    resnet101
)

box_presets=(
    t1-t5
    clavicle-only
    complete-vertebrae
)

counter=0
n_processes=8
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
