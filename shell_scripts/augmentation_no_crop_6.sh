#!/bin/bash

models=(
    resnet34
    resnet50
    resnet101
    densenet121
    densenet161
    densenet169
    densenet201
)

box_presets=(
    t1-t5
    clavicle-only
    complete-vertebrae
)

counter=0
n_processes=7
wait_time=60

# Generate the commands and pipe them to xargs
for model in "${models[@]}"; do
    for preset in "${box_presets[@]}"; do
        if [ $counter -lt $n_processes ]; then
            sleep_duration=$((counter * $wait_time))
            echo "sleep $sleep_duration && python train_base.py --model_arch $model --box_preset $preset --experiment_name augmentation_no_crop"
        else
            echo "python train_base.py --model_arch $model --box_preset $preset --experiment_name augmentation_no_crop"
        fi
        ((counter++))
    done
done | xargs -I CMD -P $n_processes bash -c CMD
