MODEL_NAME="f-actor-nanocodec-utt-align"
OUTPUT_DIR=OUTPUT_DIR_NEEDS_TO_BE_DEFINED/$MODEL_NAME
MODEL_ID=meta-llama/Llama-3.2-1B-Instruct
NUM_GPUS=4

export WANDB_PROJECT="f-actor"
export WANDB_ENTITY="TO_BE_DEFINED"

accelerate launch \
	--config-file confs/accelerate.yaml \
	--deepspeed-config-file confs/zero2.json \
  --num_processes=$NUM_GPUS \
  training/finetune.py \
  --model_name $MODEL_NAME \
  --model_id ${MODEL_ID} \
  --speech_path "maikezu/f-actor-behavior-sd-nanocodec" \
  --output_dir $OUTPUT_DIR \
  --num_dsus 4 \
  --text_stream \
  --n_delay_audio_stream 2 \
  --use_speaker_embedding \
  --use_system_narrative \
  --max_length 2048 \
  --audio_vocab_size 4032
