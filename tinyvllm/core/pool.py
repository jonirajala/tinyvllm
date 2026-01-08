"""Object pool for reusing Request, Sequence, and SchedulerOutput objects."""

from typing import Callable, Generic, List, TypeVar

T = TypeVar('T')


class ObjectPool(Generic[T]):
    """Reusable object pool to reduce allocation overhead."""

    def __init__(self, factory: Callable[[], T], max_size: int = 64):
        self._pool: List[T] = []
        self._factory = factory
        self._max_size = max_size

    def acquire(self) -> T:
        """Get object from pool or create new."""
        return self._pool.pop() if self._pool else self._factory()

    def release(self, obj: T) -> None:
        """Return object to pool after reset()."""
        if len(self._pool) < self._max_size: self._pool.append(obj)

    def clear(self) -> None:
        """Clear all pooled objects."""
        self._pool.clear()

    def size(self) -> int:
        """Get number of objects currently in pool."""
        return len(self._pool)
