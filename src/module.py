from typing import Any


class Module:
    def __init__(self, name):
        self.name = name

    def __call__(self, **kwargs: Any) -> Any:
        pass