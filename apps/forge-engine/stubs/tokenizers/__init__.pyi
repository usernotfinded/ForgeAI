from typing import Any, Iterable, Sequence

class Encoding:
    ids: list[int]

class Tokenizer:
    pre_tokenizer: Any
    decoder: Any
    post_processor: Any
    def __init__(self, model: Any) -> None: ...
    def encode(self, text: str) -> Encoding: ...
    def decode(self, ids: Sequence[int]) -> str: ...
    def token_to_id(self, token: str) -> int | None: ...
    def get_vocab_size(self) -> int: ...
    def train(self, files: Sequence[str], trainer: Any) -> None: ...
    def train_from_iterator(self, iterator: Iterable[str], trainer: Any) -> None: ...
    def save(self, path: str) -> None: ...
    @classmethod
    def from_file(cls, path: str) -> Tokenizer: ...

class _Models:
    class BPE:
        def __init__(self, unk_token: str | None = ...) -> None: ...

class _Trainers:
    class BpeTrainer:
        def __init__(
            self,
            vocab_size: int = ...,
            min_frequency: int = ...,
            special_tokens: list[str] | None = ...,
            show_progress: bool = ...,
        ) -> None: ...

class _PreTokenizers:
    class ByteLevel:
        def __init__(self, add_prefix_space: bool = ...) -> None: ...

class _Decoders:
    class ByteLevel:
        def __init__(self) -> None: ...

class _Processors:
    class ByteLevel:
        def __init__(self, trim_offsets: bool = ...) -> None: ...

models: _Models
trainers: _Trainers
pre_tokenizers: _PreTokenizers
decoders: _Decoders
processors: _Processors

def __getattr__(name: str) -> Any: ...
