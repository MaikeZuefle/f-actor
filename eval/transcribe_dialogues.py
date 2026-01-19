import torch
import torchaudio


def tuple_to_string(start_end_tuple, ndigits=1):
    return str((round(start_end_tuple[0], ndigits), round(start_end_tuple[1], ndigits)))


def format_as_transcription(segments, no_timestamps=False):
    """
    segments: list of dicts with keys 'speaker', 'start', 'end', 'text'
    """
    if no_timestamps:
        trnascript = [f"{chunk['speaker']}: {chunk['text']}" for chunk in segments]
    else:
        [
            f"{chunk['speaker']} {tuple_to_string((chunk['start'], chunk['end']))}: {chunk['text']}"
            for chunk in segments
        ]
    return "\n".join(trnascript)


def transcribe_dial(sample, asr_model, no_timestamps=False):
    """
    sample: list of audio file paths, each corresponding to a speaker channel
    asr_model: ASR model with .transcribe method returning timestamps
    """
    all_segments = []

    for idx, channel in enumerate(sample):
        speaker_label = f"Speaker {idx+1}"

        # Transcribe
        transcript = asr_model.transcribe(channel, timestamps=True)

        # Extract segments and add speaker info
        for seg in transcript[0].timestamp["segment"]:
            all_segments.append(
                {
                    "speaker": speaker_label,
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["segment"],
                }
            )

    # Sort all segments by start time
    all_segments.sort(key=lambda x: x["start"])

    # Format as readable dialogue
    output = format_as_transcription(all_segments, no_timestamps=no_timestamps)
    return output
