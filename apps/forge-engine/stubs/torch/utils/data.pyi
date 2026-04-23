from typing import Any, Generic, Iterator, TypeVar

T = TypeVar("T")

class Dataset(Generic[T]):
    def __getitem__(self, index: int) -> T: ...
    def __len__(self) -> int: ...

class DataLoader(Generic[T]):
    def __init__(
        self,
        dataset: Dataset[T],
        batch_size: int = ...,
        shuffle: bool = ...,
        num_workers: int = ...,
        pin_memory: bool = ...,
        drop_last: bool = ...,
    ) -> None: ...
    def __iter__(self) -> Iterator[T]: ...
    def __len__(self) -> int: ...


def __getattr__(name: str) -> Any: ...
