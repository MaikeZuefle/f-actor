def format_behavior(name, num_bc, num_inter):
    return f"- backchannels: {num_bc}\n- interruptions: {num_inter}"
    # return f"- backchannels: 20\n- interruptions: 20"


def build_prompt(
    example,
    max_length,
    orig_dsu_length,
    role_to_speaker_map,
    use_system_narrative=False,
    speech=False,
):

    first_speaker, num_bc_per_role, num_interrupt_per_role = get_meta_data(
        example,
        max_length=max_length,
        n_dsu=orig_dsu_length,
        role_to_speaker_map=role_to_speaker_map,
    )

    # determine system + user name
    system_id = role_to_speaker_map["system"]
    user_id = role_to_speaker_map["user"]
    system_speaker = example["speakers"][system_id]
    user_speaker = example["speakers"][user_id]

    # get system behaviour
    system_bc = num_bc_per_role["system"]
    system_interrupt = num_interrupt_per_role["system"]
    system_speaks_first = first_speaker == "system"

    # get narrative
    if use_system_narrative:
        description = "Narrative"
        narrative = f'- {example[f"new_narrative_s{system_id+1}"]}'
    else:
        description = "Narrative"
        narrative = f'- {example["narrative"]}'

    prompt = f"""Generate a dialogue between you ({system_speaker}) and another speaker ({user_speaker}) based on the given {description.lower()}. Follow the specific behavior instructions for you.

{description}:
{narrative}

Your behaviors:
{format_behavior(system_speaker, system_bc, system_interrupt)}
- starts the dialogue: {system_speaks_first}

Ensure that the dialogue reflects the behaviours of you.\n"""
    if not speech:
        prompt += "<|SOT|>"
    else:
        prompt += "<|SOS|>"

    return prompt


def in_audio_subset(utt, words, audio_duration):
    if words:
        end_utt_time = words[-1]["start"] + words[-1]["dur"]
    else:
        end_utt_time = utt["end_time"]
    return end_utt_time <= audio_duration


def get_meta_data(example, max_length, n_dsu, role_to_speaker_map):
    speaker_to_role_map = {v: k for k, v in role_to_speaker_map.items()}

    audio_duration = example["utterances"][-1]["end_time"]  # total time of conversation
    frames_per_sec = n_dsu / audio_duration
    audio_subset_duration = (
        max_length / frames_per_sec
    )  # time of conversation after max_length

    first_speaker = speaker_to_role_map[example["utterances"][0]["speaker_idx"]]

    num_bc_per_role = {"system": 0, "user": 0}
    num_interrupt_per_role = {"system": 0, "user": 0}

    for utt in example["utterances"]:

        main_speaker = speaker_to_role_map[utt["speaker_idx"]]
        other_speaker = speaker_to_role_map[1 - utt["speaker_idx"]]

        if not in_audio_subset(utt, utt["words"], audio_subset_duration):
            break

        if utt["uttr_type"] == "interrupt":
            num_interrupt_per_role[main_speaker] += 1

        backchannels = utt.get("backchannels", [])
        if backchannels:
            for bc in backchannels:
                if not in_audio_subset(bc, bc["words"], audio_subset_duration):
                    break
                num_bc_per_role[other_speaker] += 1

    return first_speaker, num_bc_per_role, num_interrupt_per_role
