MODEL_NAME="f-actor-mimi-behavior-sd-depthtransformer"
OUTPUT_DIR=OUTPUT_DIR_NEEDS_TO_BE_DEFINED/$MODEL_NAME
MODEL_ID=meta-llama/Llama-3.2-1B-Instruct
NUM_GPUS=4

accelerate launch \
  --config-file confs/accelerate.yaml \
  --deepspeed-config-file confs/zero2.json \
  --num_processes=$NUM_GPUS \
  training/finetune.py \
  --model_name $MODEL_NAME \
  --model_id ${MODEL_ID} \
  --speech_path "maikezu/f-actor-behavior-sd-mimi" \
  --preprocessing_num_workers 8 \
  --output_dir $OUTPUT_DIR \
  --gradient_accumulation_steps 8 \
  --train_batch_size 1 \
  --eval_batch_size 8 \
  --learning_rate 1e-4 \
  --text_padding_weight 0.5 \
  --silence_pad_weight 0.5 \
  --first_codebook_weight 100 \
  --num_dsus 8 \
  --text_stream \
  --n_delay_audio_stream 2 \
  --use_depth_decoder \
  --use_speaker_embedding \
  --depth_decoder_use_speaker_embedding \
  --calc_loss_on_c1_only \
  --word_alignment \
  --add_epad_token \
  --max_length 2048 \
  --audio_vocab_size 2049
