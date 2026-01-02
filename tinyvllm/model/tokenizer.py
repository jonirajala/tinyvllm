"""Tokenizer wrapper for LLaMA models."""

from pathlib import Path
from typing import List, Optional, Union
import json
import struct


class BPETokenizer:
    """
    Simple BPE tokenizer for LLaMA models.
    Supports loading from tokenizer.json (HuggingFace format) or tokenizer.model (SentencePiece).
    """

    def __init__(
        self,
        vocab: dict[str, int],
        merges: List[tuple[str, str]],
        special_tokens: dict[str, int] = None,
    ):
        self.vocab = vocab
        self.vocab_inv = {v: k for k, v in vocab.items()}
        self.merges = {pair: i for i, pair in enumerate(merges)}
        self.special_tokens = special_tokens or {}
        self.special_tokens_inv = {v: k for k, v in self.special_tokens.items()}

        # Common special token IDs
        self.bos_id = self.special_tokens.get("<s>", self.special_tokens.get("<|begin_of_text|>", 1))
        self.eos_id = self.special_tokens.get("</s>", self.special_tokens.get("<|end_of_text|>", 2))
        self.pad_id = self.special_tokens.get("<pad>", 0)

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> List[int]:
        """Encode text to token IDs."""
        # Handle special tokens first
        for special, id in self.special_tokens.items():
            if special in text:
                parts = text.split(special)
                result = []
                for i, part in enumerate(parts):
                    if part:
                        result.extend(self._encode_piece(part))
                    if i < len(parts) - 1:
                        result.append(id)
                tokens = result
                break
        else:
            tokens = self._encode_piece(text)

        if add_bos:
            tokens = [self.bos_id] + tokens
        if add_eos:
            tokens = tokens + [self.eos_id]

        return tokens

    def _encode_piece(self, text: str) -> List[int]:
        """Encode a piece of text without special tokens."""
        # Convert to bytes and then to initial tokens
        text_bytes = text.encode("utf-8")
        tokens = [self.vocab.get(bytes([b]).decode("utf-8", errors="replace"), 0) for b in text_bytes]

        # Apply BPE merges
        while len(tokens) >= 2:
            # Find the pair with lowest merge rank
            min_rank = float("inf")
            min_idx = -1

            for i in range(len(tokens) - 1):
                pair = (self.vocab_inv.get(tokens[i], ""), self.vocab_inv.get(tokens[i + 1], ""))
                if pair in self.merges:
                    rank = self.merges[pair]
                    if rank < min_rank:
                        min_rank = rank
                        min_idx = i

            if min_idx == -1:
                break

            # Merge the pair
            pair = (self.vocab_inv.get(tokens[min_idx], ""), self.vocab_inv.get(tokens[min_idx + 1], ""))
            merged = pair[0] + pair[1]
            merged_id = self.vocab.get(merged, tokens[min_idx])
            tokens = tokens[:min_idx] + [merged_id] + tokens[min_idx + 2 :]

        return tokens

    def decode(self, tokens: List[int], skip_special: bool = True) -> str:
        """Decode token IDs to text."""
        pieces = []
        for token in tokens:
            if skip_special and token in self.special_tokens_inv:
                continue
            piece = self.vocab_inv.get(token, "")
            pieces.append(piece)

        text = "".join(pieces)
        # Handle byte-level encoding
        try:
            return text.encode("latin-1").decode("utf-8")
        except (UnicodeDecodeError, UnicodeEncodeError):
            return text

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


class SentencePieceTokenizer:
    """Wrapper for SentencePiece tokenizer (tokenizer.model format)."""

    def __init__(self, model_path: Path):
        try:
            import sentencepiece as spm
        except ImportError:
            raise ImportError("sentencepiece not installed. Run: pip install sentencepiece")

        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(str(model_path))

        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        self.pad_id = self.sp.pad_id() if self.sp.pad_id() >= 0 else 0

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> List[int]:
        """Encode text to token IDs."""
        tokens = self.sp.Encode(text)
        if add_bos:
            tokens = [self.bos_id] + tokens
        if add_eos:
            tokens = tokens + [self.eos_id]
        return tokens

    def decode(self, tokens: List[int], skip_special: bool = True) -> str:
        """Decode token IDs to text."""
        if skip_special:
            tokens = [t for t in tokens if t not in (self.bos_id, self.eos_id, self.pad_id)]
        return self.sp.Decode(tokens)

    def decode_token(self, token: int) -> str:
        """Decode a single token, preserving space prefix."""
        if token in (self.bos_id, self.eos_id, self.pad_id):
            return ""
        piece = self.sp.IdToPiece(token)
        # SentencePiece uses ▁ (U+2581) to represent space
        return piece.replace("▁", " ")

    @property
    def vocab_size(self) -> int:
        return self.sp.GetPieceSize()


class TiktokenTokenizer:
    """Wrapper for tiktoken (used by newer LLaMA models)."""

    def __init__(self, encoding_name: str = "cl100k_base"):
        try:
            import tiktoken
            self.enc = tiktoken.get_encoding(encoding_name)
        except ImportError:
            raise ImportError("tiktoken not installed. Run: pip install tiktoken")

        self.bos_id = self.enc.encode("<|begin_of_text|>", allowed_special="all")[0] if "<|begin_of_text|>" in self.enc._special_tokens else 128000
        self.eos_id = self.enc.encode("<|end_of_text|>", allowed_special="all")[0] if "<|end_of_text|>" in self.enc._special_tokens else 128001
        self.pad_id = 0

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> List[int]:
        """Encode text to token IDs."""
        tokens = self.enc.encode(text, allowed_special="all")
        if add_bos:
            tokens = [self.bos_id] + tokens
        if add_eos:
            tokens = tokens + [self.eos_id]
        return tokens

    def decode(self, tokens: List[int], skip_special: bool = True) -> str:
        """Decode token IDs to text."""
        if skip_special:
            tokens = [t for t in tokens if t not in (self.bos_id, self.eos_id, self.pad_id)]
        return self.enc.decode(tokens)

    @property
    def vocab_size(self) -> int:
        return self.enc.n_vocab


def load_tokenizer(path: Union[str, Path]) -> Union[BPETokenizer, SentencePieceTokenizer, TiktokenTokenizer]:
    """
    Load a tokenizer from a file or directory.

    Supports:
    - tokenizer.json (HuggingFace format)
    - tokenizer.model (SentencePiece format)
    - Directory containing either
    """
    path = Path(path)

    if path.is_dir():
        # Check for tokenizer files - prefer SentencePiece for LLaMA models
        if (path / "tokenizer.model").exists():
            return SentencePieceTokenizer(path / "tokenizer.model")
        elif (path / "tokenizer.json").exists():
            return _load_hf_tokenizer(path / "tokenizer.json")
        else:
            raise FileNotFoundError(f"No tokenizer found in {path}")

    elif path.suffix == ".json":
        return _load_hf_tokenizer(path)
    elif path.suffix == ".model":
        return SentencePieceTokenizer(path)
    else:
        raise ValueError(f"Unsupported tokenizer format: {path}")


def _load_hf_tokenizer(path: Path) -> BPETokenizer:
    """Load HuggingFace tokenizer.json format."""
    with open(path) as f:
        data = json.load(f)

    # Extract vocab
    vocab = {}
    if "model" in data and "vocab" in data["model"]:
        vocab = data["model"]["vocab"]
    elif "vocab" in data:
        vocab = data["vocab"]

    # Extract merges
    merges = []
    if "model" in data and "merges" in data["model"]:
        for merge in data["model"]["merges"]:
            if isinstance(merge, str):
                parts = merge.split()
                if len(parts) == 2:
                    merges.append(tuple(parts))
            elif isinstance(merge, list) and len(merge) == 2:
                merges.append(tuple(merge))

    # Extract special tokens
    special_tokens = {}
    if "added_tokens" in data:
        for token in data["added_tokens"]:
            if isinstance(token, dict):
                special_tokens[token["content"]] = token["id"]

    return BPETokenizer(vocab, merges, special_tokens)
