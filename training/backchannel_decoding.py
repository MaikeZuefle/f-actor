"""Constrained decoding for forced backchannels during generate().

BackchannelLogitsProcessor forces a `[BC]` marker onto the text stream once
`bc_head`'s probability crosses a threshold, then walks a prefix trie/DFA
built from a predefined phrase list so the actual content is still sampled
from the real LM head, but restricted to a legal continuation of one of the
candidate phrases - laying down [WORD_PAD]/[EPAD]/[EOU] the same way
training data does (see dialogue_creation/get_text_stream.py). A forced
token is just the degenerate case of the same masking mechanism (an allowed
set of size 1), so there's no separate hard-override code path anywhere.

NullLogitsProcessor is the no-op counterpart, so generate() never has to
branch on whether the feature is enabled.
"""

from typing import Optional, Protocol

import torch
import yaml

# --------------------------------------------------------------------------
# Trie / DFA over the candidate phrases
# --------------------------------------------------------------------------


class BackchannelTrie:
    """A genuine prefix trie over tokenized candidate phrases, so nodes only
    merge where candidates literally share tokens so far - never merely
    because they sit at the same word index (two different words at word
    position 0, e.g. "mhm" vs "uh huh", must NOT collapse into one node).

    candidates: List[List[List[int]]] (candidate -> word -> token ids), as
    produced by tokenize_backchannel_candidates() below.

    Construction, per candidate:
      1. Insert its content tokens word by word. After each non-final word's
         tokens, insert a literal `epad_id` edge before continuing to the
         next word (matching training data: EPAD marks "a word is about to
         start", get_text_stream.py:144-161).
      2. The node reached right after each word's content tokens ("word-done"
         node) is recorded as `terminal` (if that word is the candidate's
         last) or `continuing` (otherwise) - the same node can be both, if
         one candidate is a strict word-prefix of another (e.g. "yeah" vs
         "yeah yeah"); that's only resolvable by which of epad_id/eou_id the
         model actually samples, so it requires use_eou=True (checked below).
      3. After all candidates are inserted: for every `terminal` word-done
         node, add an `eou_id` edge (if use_eou) into a fresh end node; with
         use_eou=False, an unambiguous terminal-only node (never also
         `continuing`) is itself the end - no trailing marker, matching
         training data's own EOU-less behavior.
      4. Every word-done node also gets a WORD_PAD self-loop-as-chain, capped
         at max_word_pad_frames, each link mirroring that node's real
         (epad_id/eou_id) edges - so "hold a few frames before finishing the
         word" is available at any point up to the cap, then the model is
         forced off it since the pad edge simply disappears past the cap.
    """

    def __init__(self, candidates, epad_id, eou_id, word_pad_id, max_word_pad_frames, use_eou):
        self.epad_id = epad_id
        self.eou_id = eou_id
        self.word_pad_id = word_pad_id
        self.max_word_pad_frames = max_word_pad_frames
        self.use_eou = use_eou

        self.edges = {}  # node_id -> {token_id: node_id}
        self._next_id = 0
        self.root = self._new_node()
        self.terminal_ids = set()  # word-done nodes where >=1 candidate ends
        self.continuing_ids = set()  # word-done nodes where >=1 candidate continues

        for words in candidates:
            self._insert(words)

        ambiguous = self.terminal_ids & self.continuing_ids
        if ambiguous and not use_eou:
            raise ValueError(
                "Candidate phrases are ambiguous: some word position is both "
                "a stopping point for one candidate and a continuation point "
                "for another (one candidate is a strict word-prefix of "
                "another, e.g. 'yeah' / 'yeah yeah') - disambiguating 'stop' "
                "from 'continue' requires use_eou=True."
            )

        self.end_nodes = set()
        if use_eou:
            for node in self.terminal_ids:
                self.end_nodes.add(self._child(node, eou_id))
        else:
            self.end_nodes = set(self.terminal_ids)

        self._add_word_pad_chains()

    def _new_node(self):
        node_id = self._next_id
        self._next_id += 1
        self.edges[node_id] = {}
        return node_id

    def _child(self, node, token_id):
        children = self.edges[node]
        if token_id not in children:
            children[token_id] = self._new_node()
        return children[token_id]

    def _insert(self, words):
        node = self.root
        n = len(words)
        for word_idx, tokens in enumerate(words):
            for token_id in tokens:
                node = self._child(node, token_id)
            if word_idx == n - 1:
                self.terminal_ids.add(node)
            else:
                self.continuing_ids.add(node)
                node = self._child(node, self.epad_id)

    def _add_word_pad_chains(self):
        for base in self.terminal_ids | self.continuing_ids:
            node = base
            base_real_edges = [
                (tok, target)
                for tok, target in self.edges[base].items()
                if tok != self.word_pad_id
            ]
            for _ in range(self.max_word_pad_frames):
                pad_node = self._new_node()
                self.edges[node][self.word_pad_id] = pad_node
                for tok, target in base_real_edges:
                    self.edges[pad_node][tok] = target
                node = pad_node

    def allowed_tokens(self, node):
        return self.edges.get(node, {})

    def edge(self, node, token_id):
        return self.edges.get(node, {}).get(token_id)

    def is_terminal(self, node):
        return node in self.end_nodes


def tokenize_backchannel_candidates(tokenizer, phrases):
    """Tokenize each phrase into per-word token-id lists, matching the
    lower-casing + leading-space-per-word-after-first convention used to
    build training data (dialogue_creation/get_text_stream.py:125-138), so
    the injected sequence matches what the model was trained to expect
    after [BC].
    """
    candidates = []
    for phrase in phrases:
        words = phrase.lower().split()
        word_tokens = []
        for wi, word in enumerate(words):
            text = word if wi == 0 else " " + word
            tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
            word_tokens.append(tokens)
        candidates.append(word_tokens)
    return candidates


def load_backchannel_phrases(path):
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data["backchannels"]


# --------------------------------------------------------------------------
# Logits processors
# --------------------------------------------------------------------------

FREE = "FREE"
FORCED = "FORCED"
COOLDOWN = "COOLDOWN"

NEG_INF = float("-inf")


class LogitsProcessor(Protocol):
    def process(
        self,
        ts_logits: torch.Tensor,
        bc_probs: Optional[torch.Tensor],
        prev_tokens: Optional[torch.Tensor],
    ) -> torch.Tensor: ...


class NullLogitsProcessor:
    """No-op: generate() never has to branch on whether BC forcing is on."""

    def process(self, ts_logits, bc_probs, prev_tokens):
        return ts_logits


class BackchannelLogitsProcessor:
    def __init__(
        self,
        tokenizer,
        phrases,
        threshold,
        cooldown_steps,
        bc_token_id,
        epad_id,
        eou_id,
        word_pad_id,
        inter_token_id=None,
        silence_pad_id=None,
        max_word_pad_frames=6,
        use_eou=True,
        batch_size=1,
    ):
        candidates = tokenize_backchannel_candidates(tokenizer, phrases)
        self.trie = BackchannelTrie(
            candidates, epad_id, eou_id, word_pad_id, max_word_pad_frames, use_eou
        )

        self.threshold = threshold
        self.cooldown_steps = cooldown_steps
        self.bc_token_id = bc_token_id
        self.inter_token_id = inter_token_id
        # If set, force [SILENCE_PAD] (rather than letting the LM head
        # sample freely) for the whole cooldown window after a forced
        # backchannel completes - matching the token training data uses for
        # "no speech content here" (get_text_stream.py's add_audio_delay),
        # so the model doesn't immediately start another utterance right
        # after being forced to backchannel. None preserves the old
        # behavior (free sampling, [BC]/[INTER] masked, during cooldown).
        self.silence_pad_id = silence_pad_id

        B = batch_size
        self.mode = [FREE] * B
        self.trie_state = [None] * B
        self.skip_next_advance = [False] * B
        self.cooldown_remaining = [0] * B

    def process(self, ts_logits, bc_probs, prev_tokens):
        self._update_state(prev_tokens, bc_probs)
        return self._process_logits(ts_logits)

    def _update_state(self, prev_tokens, bc_probs):
        B = len(self.mode)
        for b in range(B):
            if self.mode[b] == FORCED:
                if self.skip_next_advance[b]:
                    # this frame's prev token was the [BC] entry marker
                    # itself, not a trie edge - nothing to advance.
                    self.skip_next_advance[b] = False
                else:
                    token_id = int(prev_tokens[b])
                    self.trie_state[b] = self.trie.edge(self.trie_state[b], token_id)
                    if self.trie.is_terminal(self.trie_state[b]):
                        self.mode[b] = COOLDOWN
                        self.cooldown_remaining[b] = self.cooldown_steps
            elif self.mode[b] == COOLDOWN:
                self.cooldown_remaining[b] -= 1
                if self.cooldown_remaining[b] <= 0:
                    self.mode[b] = FREE

            if (
                self.mode[b] == FREE
                and bc_probs is not None
                and float(bc_probs[b]) > self.threshold
            ):
                self.mode[b] = FORCED
                self.trie_state[b] = self.trie.root
                self.skip_next_advance[b] = True

    def _process_logits(self, ts_logits):
        B = len(self.mode)
        for b in range(B):
            if self.mode[b] == FORCED and self.skip_next_advance[b]:
                allowed = {self.bc_token_id}
            elif self.mode[b] == FORCED:
                allowed = self.trie.allowed_tokens(self.trie_state[b])
            elif self.mode[b] == COOLDOWN and self.silence_pad_id is not None:
                allowed = {self.silence_pad_id}
            else:  # FREE (no trigger), or COOLDOWN without forced silence:
                # never let [BC]/[INTER] through
                allowed = None

            if allowed is None:
                mask = torch.zeros_like(ts_logits[b], dtype=torch.bool)
                mask[..., self.bc_token_id] = True
                if self.inter_token_id is not None:
                    mask[..., self.inter_token_id] = True
            else:
                allowed_ids = torch.tensor(
                    list(allowed), device=ts_logits.device, dtype=torch.long
                )
                keep = torch.zeros_like(ts_logits[b], dtype=torch.bool)
                keep[..., allowed_ids] = True
                mask = ~keep

            ts_logits[b] = ts_logits[b].masked_fill(mask, NEG_INF)
        return ts_logits
