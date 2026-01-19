python add_speech_tokens.py \
  --model_id google/gemma-3-270m \
  --save_path /home/jovyan/shared/abirch/work/maike/trained_models/google_gemma-3-270_new_tok \
  --set_pad_to_eos \
  --new_tokens '["[laughter]", "<laughter>", "</laughter>", "<OVERLAP>", "<BC_S>", "<BC_E>", "<GAP>", "<LISTEN>", "<SPEAK>"]' \
  --new_special_tokens '["<|SOT|>", "<|SOS|>", "<|EOS|>"]' \
  --number_audio_tokens 2050
