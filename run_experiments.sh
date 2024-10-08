#!/bin/bash

GPU="0"
datasets=("periodic" "burst" "bipartite" "triadic")
models=("JODIE" "DyRep" "TGAT" "TGN" "CAWN" "TCL" "GraphMixer" "DyGFormer")
sampling=("random" "historical" "inductive")
shuffle_orders=("" "--shuffle_order")
arange_timestamps=("" "--arange_timestamp")

for dataset in "${datasets[@]}"
do
    for model in "${models[@]}"
    do
        for shuffle_order in "${shuffle_orders[@]}"
        do
            for arange_timestamp in "${arange_timestamps[@]}"
            do
                for sample_strategy in "${sampling[@]}"
                do
                    python train_link_prediction.py --dataset_name $dataset --model_name $model --patience 5 --load_best_configs --num_runs 5 --gpu $GPU --negative_sample_strategy $sample_strategy $shuffle_order $arange_timestamp
                done
            done
        done
    done
done

for dataset in "${datasets[@]}"
do
    for shuffle_order in "${shuffle_orders[@]}"
    do
        for arange_timestamp in "${arange_timestamps[@]}"
        do
            for sample_strategy in "${sampling[@]}"
            do
                python evaluate_link_prediction.py --dataset_name $dataset --model_name EdgeBank --patience 5 --load_best_configs --num_runs 5 --gpu $GPU --negative_sample_strategy $sample_strategy $shuffle_order $arange_timestamp
            done
        done
    done
done
