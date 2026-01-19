#!/bin/bash

source ~/.bashrc
conda activate codec_processing
NUM_GPUS=2
MASTER_PORT=29501
SUBSET=train
DATA_DIR=/home/jovyan/shared/abirch/work/tklam/dir_data/behavior-sd/${SUBSET}/

echo "Searching for the ${NUM_GPUS} most free GPUs..."
export CUDA_VISIBLE_DEVICES=$(python3 nanocodec_processing/get_free_gpus.py --num_gpus $NUM_GPUS)
echo "Found and set CUDA_VISIBLE_DEVICES to: ${CUDA_VISIBLE_DEVICES}"

torchrun --nproc_per_node=$NUM_GPUS \
	       --master_port=$MASTER_PORT \
           nanocodec_processing/nanocodec_preproc.py \
            --data_dir "$DATA_DIR" \
            --output_dir "nanocodec_processing/4codebook_processed_data/${SUBSET}" \
            --batch_size 1 \
            --codec_ckpt "nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps"
