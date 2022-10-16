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
        _op: str = "",
    ):
        self.data = data
        self.grad = 0.0
        self.children = set() if children is None else children
        self._backwards = _backwards
        self._op = _op

    def __add__(self, other: Union[float, Value]) -> Value:
        """self + other"""

        if isinstance(other, (float, int)):
            other = Value(other)

        def _backward(grad_parent: float) -> None:
            self.grad += grad_parent
            other.grad += grad_parent

        return Value(
            self.data + other.data,
            children={self, other},
            _backwards=_backward,
            _op="+",
        )

    def __mul__(self, other: Union[float, Value]) -> Value:
        if isinstance(other, (float, int)):
            other = Value(other)

        def _backward(grad_parent: float) -> None:
            self.grad += other.data * grad_parent
            other.grad += self.data * grad_parent

        return Value(
            self.data * other.data,
            children={self, other},
            _backwards=_backward,
            _op="*",
        )

    def relu(self) -> Value:
        def _backwards(grad_parent: float) -> None:
            local_gradient = 1.0 if self.data > 0 else 0.0
            self.grad += local_gradient * grad_parent

        output_data = 0 if self.data < 0 else self.data
        return Value(
            output_data,
            children={self},
            _backwards=_backwards,
            _op="relu",
        )

    def tanh(self) -> Value:
        def _backwards(grad_parent: float) -> None:
            local_gradient = 1 - math.tanh(self.data) ** 2
            self.grad += local_gradient * grad_parent

        return Value(
            math.tanh(self.data),
            children={self},
            _backwards=_backwards,
            _op="tanh",
        )

    def __pow__(self, power: float) -> Value:
        """self ** power"""
        assert isinstance(power, (int, float)), "Only support int/float powers"

        def _backward(grad_parent: float) -> None:
            self.grad += (power * self.data ** (power - 1)) * grad_parent

        return Value(
            self.data**power, children={self}, _backwards=_backward, _op=f"^{power}"
        )

    def backward(self) -> None:
        self.grad = 1

        # for v in find_reversed_topological_order(self):
        for v in find_reversed_topological_order(self):
            v._backwards(v.grad)

    def __neg__(self) -> Value:
        """-self"""
        return self * -1

    def __radd__(self, other: Union[Value, float]) -> Value:
        """other + self
        Python fallback when other is not a Value.
        """
        return self + other

    def __sub__(self, other: Union[Value, float]) -> Value:
        """self - other"""

        return self + (-other)

    def __rsub__(self, other: float) -> Value:
        """other - self
        Python fallback when other is not a Value.
        """
        return other + (-self)

    def __rmul__(self, other: float) -> Value:
        """other * self
        Python fallback when other is not a Value.
        """
        return self * other

    def __truediv__(self, other: float):
        """self / other"""
        return self * other**-1

    def __rtruediv__(self, other: float):
        """other / self
        Python fallback when other is not a Value.
        """
        return other * self**-1

    def __repr__(self) -> str:
        return f"Value(data={self.data})"


def find_reversed_topological_order(root: Value):
    visited = set()
    topological_order = []

    def _depth_first_search(node: Value):
        if node not in visited:
            visited.add(node)

            for child in node.children:
                _depth_first_search(child)

            topological_order.append(node)

    _depth_first_search(root)

    return list(reversed(topological_order))
