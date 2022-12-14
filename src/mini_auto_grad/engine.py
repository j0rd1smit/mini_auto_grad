from __future__ import annotations

import math
from typing import Callable, Optional, Union

Backwards = Callable[[float], None]


def _noop_grad(_: float) -> None:
    pass


class Value:
    def __init__(
        self,
        data: float,
        children: Optional[set[Value]] = None,
        _backwards: Backwards = _noop_grad,
    ):
        self.data = data
        self.grad = 0.0
        self.children = set() if children is None else children
        self._backwards = _backwards

    def __add__(self, other: Union[float, Value]) -> Value:
        """self + other"""
        if isinstance(other, (float, int)):
            pass

        def _backward(grad_parent: float) -> None:
            pass  # TODO ex2

        return None  # TODO ex1

    def __mul__(self, other: Union[float, Value]) -> Value:
        if isinstance(other, (float, int)):
            other = Value(other)

        def _backward(grad_parent: float) -> None:
            pass  # TODO ex2

        return None  # TODO ex1

    def tanh(self) -> Value:
        def _backwards(grad_parent: float) -> None:
            pass

        return None  # TODO ex1

    def __pow__(self, power: float) -> Value:
        """self ** power"""
        assert isinstance(power, (int, float)), "Only support int/float powers"

        def _backward(grad_parent: float) -> None:
            pass  # TODO ex2

        return None  # TODO ex1

    def backward(self) -> None:
        self.grad = ...  # TODO ex2

        for v in find_reversed_topological_order(self):
            v._backwards(v.grad)

    def __neg__(self) -> Value:
        """-self"""
        return None  # TODO ex1

    def __radd__(self, other: Union[Value, float]) -> Value:
        """other + self
        Python fallback when other is not a Value.
        """
        return None  # TODO ex1

    def __sub__(self, other: Union[Value, float]) -> Value:
        """self - other"""

        return None  # TODO ex1

    def __rsub__(self, other: Union[float, Value]) -> Value:
        """other - self
        Python fallback when other is not a Value.
        """
        return None  # TODO ex1

    def __rmul__(self, other: Union[float, Value]) -> Value:
        """other * self
        Python fallback when other is not a Value.
        """
        return None  # TODO ex1

    def __truediv__(self, other: Union[float, Value]):
        """self / other"""
        return None  # TODO

    def __rtruediv__(self, other: Union[float, Value]):
        """other / self
        Python fallback when other is not a Value.
        """
        return None  # TODO ex1

    def __repr__(self) -> str:
        return f"Value(data={self.data})"


def find_reversed_topological_order(root: Value):
    visited = set()
    topological_order = []

    def _depth_first_search(node: Value):
        pass  # TODO ex2

    _depth_first_search(root)

    return list(reversed(topological_order))
