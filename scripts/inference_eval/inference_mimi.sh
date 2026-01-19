GENERATE=true
CONVERT_TO_AUDIO=true


# directories
MODEL_DIR=MODEL_DIR_NEEDS_TO_BE_DEFINED/"f-actor-mimi"
INFERENCE_DIR=NEEDS_TO_BE_DEFINED

# model
NUM_DSUS=8
TEXT_STREAM=true
N_DELAY_AUDIO_STREAM=2
WORD_ALIGNMENT=true

# inference
MAX_LENGTH=1024
INFERENCE_SUBSET=100
DO_SAMPLE=true
TEMPERATURE=0.6
TOP_K=40
TOP_P=1.0
USE_SPEAKER_SAMPLE=0
TALK_TO_ITSELF=true
USE_SPEAKER_EMBEDDING=true
ADD_BC_TOKEN=true
USE_SYSTEM_NARRATIVE=true

if [ "$TALK_TO_ITSELF" = true ]; then
    INFERENCE_DIR="${INFERENCE_DIR}talk_to_itself"
fi

# data
DATASET_PATH="maikezu/f-actor-behavior-sd-mimi"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# --- Helper to include flags conditionally ---
flag_if_true() {
    [ "$1" = true ] && echo "$2"
}

# --- Run inference ---
if [ "$GENERATE" = true ]; then
    echo "Running inference / generation..."
    python training/inference.py \
      --model_id "$MODEL_DIR" \
      --speech_path "$DATASET_PATH" \
      --inf_output_dir "$INFERENCE_DIR" \
      --max_length "$MAX_LENGTH" \
      --inference_on_subset "$INFERENCE_SUBSET" \
      --num_dsus "$NUM_DSUS" \
      $(flag_if_true $DO_SAMPLE "--do_sample") \
      --temperature "$TEMPERATURE" \
      --top_k "$TOP_K" \
      --top_p "$TOP_P" \
      $(flag_if_true $TEXT_STREAM "--text_stream") \
      $(flag_if_true $WORD_ALIGNMENT "--word_alignment") \
      --n_delay_audio_stream $N_DELAY_AUDIO_STREAM \
      --use_speaker_sample "$USE_SPEAKER_SAMPLE" \
      $(flag_if_true $TALK_TO_ITSELF "--talk_to_itself") \
      $(flag_if_true $USE_SPEAKER_EMBEDDING "--use_speaker_embedding") \
      $(flag_if_true $USE_SYSTEM_NARRATIVE "--use_system_narrative") \
      $(flag_if_true $ADD_BC_TOKEN "--add_bc_token")
fi

# --- Read / convert outputs ---
if [ "$CONVERT_TO_AUDIO" = true ]; then
    echo "Reading outputs / converting to audio..."
    python training/convert_dsu_to_audio.py \
      --inf_output_dir "$INFERENCE_DIR" \
      --num_dsus "$NUM_DSUS" \
      $(flag_if_true $DO_SAMPLE "--do_sample") \
      --temperature "$TEMPERATURE" \
      --top_k "$TOP_K" \
      --top_p "$TOP_P"
fi
