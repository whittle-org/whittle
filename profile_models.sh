#!/bin/bash

for strategy in sandwich standard random
do
    for compile in True False
    do
        python whittle/pretrain_super_network.py EleutherAI/pythia-14m --data TinyStories --data.data_path ./data --tokenizer_dir ~/checkpoints/EleutherAI/pythia-14m/ --out_dir pretrained_super_net --train.save_interval 1 --train.max_tokens 100000 --training_strategy $strategy --compile_model $compile
        mv trace.json trace_strategy_${strategy}_compile_${compile}.json
    done
done

