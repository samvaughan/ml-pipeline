from typing import Any, Callable, Dict


class Registry:
    def __init__(self, kind: str):
        self.kind = kind
        self._items: Dict[str, Callable[..., Any]] = {}

    def register(self, name: str):
        def deco(obj: Callable[..., Any]):
            if name in self._items:
                raise KeyError(f"{self.kind} '{name}' already registered")
            self._items[name] = obj
            return obj

        return deco

    def get(self, name: str) -> Callable[..., Any]:
        if name not in self._items:
            opts = ", ".join(sorted(self._items))
            raise KeyError(f"Unknown {self.kind} '{name}'. Available: {opts}")
        return self._items[name]

    def create(self, name: str, **kwargs):
        return self.get(name)(**kwargs)

    def names(self):
        return sorted(self._items)
