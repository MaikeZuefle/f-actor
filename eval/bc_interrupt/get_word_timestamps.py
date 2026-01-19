import json
import wave

from silero_vad import get_speech_timestamps, read_audio


def get_kaldi_timestps(wav_path, kaldi_model, return_words_tmsp=False):
    from vosk import KaldiRecognizer

    wf = wave.open(wav_path, "rb")
    rec = KaldiRecognizer(kaldi_model, wf.getframerate())
    rec.SetWords(True)
    rec.SetPartialWords(True)

    word_times = []
    segment_times = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break

        if rec.AcceptWaveform(data):
            utterance = json.loads(rec.Result())
            text = utterance["text"]
            if text:
                segment_times.append(
                    {
                        "segment": text,
                        "start": utterance["result"][0]["start"],
                        "end": utterance["result"][-1]["end"],
                    }
                )
                for entry in utterance["result"]:
                    start = entry["start"]
                    end = entry["end"]
                    word = entry["word"]
                    word_times.append({"word": word, "start": start, "end": end})

    if return_words_tmsp:
        return word_times
    return segment_times


def get_parakeet_timestps(
    wav_path, asr_model, return_words_tmsp=False, return_both=False
):
    transcript = asr_model.transcribe(wav_path, timestamps=True)
    words = transcript[0].timestamp["word"]
    segments = transcript[0].timestamp["segment"]
    if return_both:
        return words, segments
    if return_words_tmsp:
        return words
    return segments


def get_silero_timestps(wav_path, model):
    wav = read_audio(wav_path)
    speech_timestamps = get_speech_timestamps(
        wav,
        model,
        return_seconds=True,
        threshold=0.5,
        min_silence_duration_ms=1500,
    )

    return speech_timestamps
