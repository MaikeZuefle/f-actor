#!/bin/bash

source ~/.bashrc
conda activate codec_processing
NUM_GPUS=1
MASTER_PORT=29500
SUBSET=train
DATA_DIR=/home/jovyan/shared/abirch/work/tklam/dir_data/behavior-sd/${SUBSET}/

torchrun --nproc_per_node=$NUM_GPUS \
	 --master_port=$MASTER_PORT \
         proc_behavior_sd.py \
         --data_dir "$DATA_DIR" \
         --output_dir "processed_data/${SUBSET}"
#python proc_behavior_sd.py --data_dir $DATA_DIR --output_dir processed_data/train
