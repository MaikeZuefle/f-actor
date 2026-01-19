SILENCE_PAD = "[SILENCE_PAD]"
UTTERANCE_PAD = "[UTTERANCE_PAD]"
WORD_PAD = "[WORD_PAD]"
BC_TOKEN = "[BC]"
INTER_TOKEN = "[INTER]"

BC_COUNTS = {
    1: "[BC1]",
    2: "[BC2]",
    3: "[BC3]",
    4: "[BC4]",
    5: "[BC5]",
    6: "[BC6]",
    7: "[BC7]",
    8: "[BC8]",
    9: "[BC9]",
    10: "[BC10]",
    11: "[BC11]",
    12: "[BC12]",
    13: "[BC13]",
    14: "[BC14]",
    15: "[BC15]",
    16: "[BC16]",
    17: "[BC17]",
    18: "[BC18]",
    19: "[BC19]",
    20: "[BC20]",  #
    21: "[BC21]",
    22: "[BC22]",
    23: "[BC23]",
    24: "[BC24]",
    25: "[BC25]",
    26: "[BC26]",
    27: "[BC27]",
    28: "[BC28]",
    29: "[BC29]",
    30: "[BC30]",
    31: "[BC31]",
    32: "[BC32]",
    33: "[BC33]",
    34: "[BC34]",
    35: "[BC35]",
    36: "[BC36]",
    37: "[BC37]",
    38: "[BC38]",
    39: "[BC39]",
    40: "[BC40]",
}

INTER_COUNTS = {
    1: "[INTERRUPT1]",
    2: "[INTERRUPT2]",
    3: "[INTERRUPT3]",
    4: "[INTERRUPT4]",
    5: "[INTERRUPT5]",
    6: "[INTERRUPT6]",
    7: "[INTERRUPT7]",
    8: "[INTERRUPT8]",
    9: "[INTERRUPT9]",
    10: "[INTERRUPT10]",
    11: "[INTERRUPT11]",
    12: "[INTERRUPT12]",
    13: "[INTERRUPT13]",
    14: "[INTERRUPT14]",
    15: "[INTERRUPT15]",
    16: "[INTERRUPT16]",
    17: "[INTERRUPT17]",
    18: "[INTERRUPT18]",
    19: "[INTERRUPT19]",
    20: "[INTERRUPT20]",
}

BC_COUNTS_TOKENS = list(BC_COUNTS.values())
INTER_COUNTS_TOKENS = list(INTER_COUNTS.values())

TEXT_STREAM_TOKENS = (
    [SILENCE_PAD, UTTERANCE_PAD, WORD_PAD, BC_TOKEN, INTER_TOKEN]
    + INTER_COUNTS_TOKENS
    + BC_COUNTS_TOKENS
)
