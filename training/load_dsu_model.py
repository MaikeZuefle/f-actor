import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import load_file


class ModelInitializerLoader:
    def init_or_load_audio_heads(self, model_path=None):
        if self.num_dsus < 1:
            return

        if self.checkpoint_has_weights(model_path, "dsu_head"):
            self.load_dsu_heads(model_path)
        else:
            self.init_dsu_heads()

    def init_or_load_audio_embeds(self, model_path=None):
        if self.num_dsus < 1:
            return

        if self.checkpoint_has_weights(model_path, "audio_embeds.0"):
            self.load_audio_embeds(model_path)
        else:
            self.init_audio_embeds()

    def init_or_load_text_heads(self, model_path=None):
        if not self.multi_text_stream:
            return

        if self.checkpoint_has_weights(model_path, "text_head.0.weight"):
            self.load_text_heads(model_path)
        else:
            self.init_text_heads()

    def init_or_load_speaker_embed_proj(self, model_path=None):
        """
        Initialize or load the speaker embedding projection layer.
        Projects 192-dim speaker embeddings into model hidden size.
        """
        if not self.use_speaker_embedding:
            return

        # Speaker embedding input dimension
        spk_emb_dim = 192

        if self.checkpoint_has_weights(model_path, "speaker_embed_proj.weight"):
            self.load_speaker_embed_proj(model_path, spk_emb_dim)
        else:
            self.init_speaker_embed_proj(spk_emb_dim)

    def checkpoint_has_weights(self, model_path, key_prefix):
        """
        Check if the checkpoint (single or sharded safetensors) contains any
        tensor whose name starts with key_prefix.
        """
        # Case 1: sharded checkpoint with index.json
        index_file = os.path.join(model_path, "model.safetensors.index.json")
        if os.path.isfile(index_file):
            with open(index_file, "r") as f:
                index = json.load(f)
            return any(
                tensor_name.startswith(key_prefix)
                for tensor_name in index.get("weight_map", {}).keys()
            )

        # Case 2: single safetensors file
        st_file = os.path.join(model_path, "model.safetensors")
        if os.path.isfile(st_file):
            state_dict = load_file(st_file)
            return any(
                tensor_name.startswith(key_prefix) for tensor_name in state_dict.keys()
            )

        # No checkpoint found
        return False

    def init_dsu_heads(self):
        """Create DSU heads"""
        if self.num_dsus < 1:
            return

        self.num_dsu_heads = self.num_dsus * 2
        self.dsu_head = torch.nn.Linear(
            self.hidden_size, self.audio_vocab_size * self.num_dsu_heads
        )

    def init_text_heads(self):
        """
        Initialize two heads for multi_text_stream.
        """
        if not self.multi_text_stream:
            return

        self.num_text_heads = 2

        self.text_head = torch.nn.ModuleList(
            [
                torch.nn.Linear(self.hidden_size, self.text_vocab_size).to(self.device)
                for _ in range(self.num_text_heads)
            ]
        )

        for head in self.text_head:
            torch.nn.init.xavier_uniform_(head.weight)
            torch.nn.init.zeros_(head.bias)

    def init_audio_embeds(self):
        """Initialize Audio embedding layers"""
        if self.num_dsus < 1:
            return

        self.num_audio_embeds = self.num_dsu_heads

        # One embedding table per head
        self.audio_embeds = torch.nn.ModuleList(
            [
                torch.nn.Embedding(
                    num_embeddings=self.audio_vocab_size,
                    embedding_dim=self.hidden_size,
                )
                for _ in range(self.num_audio_embeds)
            ]
        )

        # Initialize embeddings
        for emb in self.audio_embeds:
            torch.nn.init.xavier_uniform_(emb.weight)

    def init_speaker_embed_proj(self, spk_emb_dim):
        """
        Initialize a linear projection from speaker embedding dimension to hidden size.
        """
        self.speaker_embed_proj = torch.nn.Linear(spk_emb_dim, self.hidden_size).to(
            self.device
        )
        torch.nn.init.xavier_uniform_(self.speaker_embed_proj.weight)
        torch.nn.init.zeros_(self.speaker_embed_proj.bias)

    def load_safetensors_state_dict(self, model_path):
        """Load a state dict from safetensors files, handling sharded or single files."""
        state_dict = {}

        index_file = os.path.join(model_path, "model.safetensors.index.json")
        if os.path.isfile(index_file):
            # Case 1: sharded checkpoint with index.json
            with open(index_file, "r") as f:
                index = json.load(f)
            for tensor_name, shard_file in index["weight_map"].items():
                shard_path = os.path.join(model_path, shard_file)
                with safe_open(shard_path, framework="pt", device="cpu") as f:
                    state_dict[tensor_name] = f.get_tensor(tensor_name)
        else:
            # Case 2: single or multiple .safetensors files
            safetensors_files = [
                f for f in os.listdir(model_path) if f.endswith(".safetensors")
            ]
            for f in safetensors_files:
                shard_path = os.path.join(model_path, f)
                shard_dict = load_file(shard_path)
                state_dict.update(shard_dict)

        return state_dict

    def load_dsu_heads(self, model_path):
        """Load DSU heads from saved safetensors (for inference)."""
        if self.num_dsus < 1:
            return

        self.num_dsu_heads = self.num_dsus * 2
        self.dsu_head = torch.nn.Linear(
            self.hidden_size, self.audio_vocab_size * self.num_dsu_heads
        )

        state_dict = self.load_safetensors_state_dict(model_path)

        # Ensure required keys exist
        for key in ["dsu_head.weight", "dsu_head.bias"]:
            if key not in state_dict:
                raise KeyError(f"Missing key '{key}' in checkpoint {model_path}")

        pretrained_weight = state_dict["dsu_head.weight"]
        pretrained_bias = state_dict["dsu_head.bias"]

        # Copy weights into the DSU head
        self.dsu_head.weight.data.copy_(pretrained_weight)
        self.dsu_head.bias.data.copy_(pretrained_bias)
        self.dsu_head.to(self.device)

    def load_text_heads(self, model_path):
        state_dict = self.load_safetensors_state_dict(model_path)
        if self.num_text_heads is None:
            self.num_text_heads = 2 if self.multi_text_stream else 1

        self.text_head = torch.nn.ModuleList(
            [
                torch.nn.Linear(self.hidden_size, self.text_vocab_size)
                for _ in range(self.num_text_heads)
            ]
        )

        for i, head in enumerate(self.text_head):
            weight_key = f"text_head.{i}.weight"
            bias_key = f"text_head.{i}.bias"
            if weight_key not in state_dict or bias_key not in state_dict:
                raise KeyError(f"Missing key {weight_key} or {bias_key} in checkpoint")
            head.weight.data.copy_(state_dict[weight_key])
            head.bias.data.copy_(state_dict[bias_key])
            head.to(self.device)

    def load_audio_embeds(self, model_path):
        """Load separate audio embeddings from saved safetensors."""
        if self.num_dsus < 1:
            return

        self.num_audio_embeds = self.num_dsus * 2

        self.audio_embeds = torch.nn.ModuleList(
            [
                torch.nn.Embedding(
                    num_embeddings=self.audio_vocab_size,
                    embedding_dim=self.hidden_size,
                )
                for _ in range(self.num_audio_embeds)
            ]
        )

        state_dict = self.load_safetensors_state_dict(model_path)

        num_pretrained_embeds = self.num_audio_embeds

        # Load weights into embeddings
        for head_idx in range(self.num_audio_embeds):
            src_idx = head_idx % num_pretrained_embeds
            weight_key = f"audio_embeds.{src_idx}.weight"
            if weight_key not in state_dict:
                raise KeyError(
                    f"Missing weight {weight_key} in checkpoint {model_path}"
                )
            self.audio_embeds[head_idx].weight.data.copy_(state_dict[weight_key])

        # Move embeddings to model's device
        for emb in self.audio_embeds:
            emb.to(self.device)

    def load_speaker_embed_proj(self, model_path, spk_emb_dim):
        """
        Load speaker embedding projection layer weights from checkpoint.
        """
        state_dict = self.load_safetensors_state_dict(model_path)

        self.speaker_embed_proj = torch.nn.Linear(spk_emb_dim, self.hidden_size)
        weight_key = "speaker_embed_proj.weight"
        bias_key = "speaker_embed_proj.bias"

        if weight_key not in state_dict or bias_key not in state_dict:
            raise KeyError(f"Missing key {weight_key} or {bias_key} in checkpoint")

        self.speaker_embed_proj.weight.data.copy_(state_dict[weight_key])
        self.speaker_embed_proj.bias.data.copy_(state_dict[bias_key])
        self.speaker_embed_proj.to(self.device)
