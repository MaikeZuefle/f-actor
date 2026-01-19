#!/bin/bash

MODELS=(
    MODEL_DIR_NEEDS_TO_BE_DEFINED/"f-actor-nanocodec-special-tokens"
    MODEL_DIR_NEEDS_TO_BE_DEFINED/"f-actor-nanocodec-system-loss-only"
)

# Judge model
JUDGE_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"

# Loop through each model
for MODEL in "${MODELS[@]}"; do
    echo "🔹 Running eval for model: $MODEL_NAME"

    MODEL_DIR="$MODEL"
    INFERENCE_DIR="$MODEL_DIR/inference_gpt"
    WAVS_DIR="$INFERENCE_DIR"/talk_to_itself/outputs_wavs*
    OUT_DIR="$INFERENCE_DIR"
    GENERATED_SAMPLES="$INFERENCE_DIR"/talk_to_itself/outputs_samples-*.json

    success=true

    python eval/utmos_scores.py \
        --wavs_dir $WAVS_DIR \
        --out_dir "$OUT_DIR" || success=false

    python eval/text_vs_dsu_stream.py \
        --wavs_dir $WAVS_DIR \
        --text_stream_json $GENERATED_SAMPLES \
        --out_dir "$OUT_DIR" || success=false

    python eval/get_stats.py \
        --wavs_dir $WAVS_DIR \
        --out_dir "$OUT_DIR" || success=false

    python eval/calc_speaker_sim.py \
        --output_dir "$OUT_DIR" \
        --wavs_dir $WAVS_DIR \
        --speaker_emb_file $GENERATED_SAMPLES || success=false

    python eval/judge_dialogues.py \
        --judge "$JUDGE_MODEL" \
        --output_dir $OUT_DIR \
        --wavs_dir $WAVS_DIR \
        --instruction_file $GENERATED_SAMPLES || success=false

    python eval/count_backchannel_interrupt.py \
        --output_dir "$OUT_DIR" \
        --wavs_dir $WAVS_DIR \
        --text_stream_json $GENERATED_SAMPLES || success=false


    if [ "$success" = true ]; then
        echo "✅ Finished evaluating $MODEL_NAME"
    else
        echo "❌ Evaluation failed for $MODEL_NAME (one or more steps failed)"
    fi

    echo "------------------------------------------------------"
done

echo "✅ Done with all judging!"
