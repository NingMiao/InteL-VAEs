#!/bin/bash
datasets=(mnist fashion_mnist)
betas=(0.3 1.0 3.0)
gammas=(10.0 30.0 100.0 300.0)

for ((i=0;i<${#gammas[@]};i++)); do
    for ((j=0;j<${#datasets[@]};j++)); do
        python downstream_classification.py --dataset ${datasets[$j]} --beta 3.0 --mapping 'sparse' --gamma ${gammas[$i]} &
    done
done