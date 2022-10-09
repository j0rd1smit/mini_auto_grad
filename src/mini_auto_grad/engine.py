from __future__ import annotations

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
        if isinstance(other, (float, int)):
            other = Value(other)

        def _backward(grad_parent: float) -> None:
            self.grad += grad_parent
            other.grad += grad_parent

        return Value(
            self.data + other.data, children={self, other}, _backwards=_backward
        )

    def __mul__(self, other: Union[float, Value]) -> Value:
        if isinstance(other, (float, int)):
            other = Value(other)

        def _backward(grad_parent: float) -> None:
            self.grad += other.data * grad_parent
            other.grad += self.data * grad_parent

        return Value(
            self.data * other.data, children={self, other}, _backwards=_backward
        )

    def relu(self):
        def _backward(grad_parent: float) -> None:
            local_gradient = 1.0 if self.data > 0 else 0.0
            self.grad += local_gradient * grad_parent

        output_data = 0 if self.data < 0 else self.data
        return Value(output_data, children={self}, _backwards=_backward)

    def __neg__(self) -> Value:
        return self * -1

    def __radd__(self, other: Union[Value, float]) -> Value:
        return self + other

    def __sub__(self, other: Union[Value, float]) -> Value:
        return self + (-other)

    def __rsub__(self, other: Union[Value, float]) -> Value:
        return other + (-self)

    def __rmul__(self, other: Union[Value, float]) -> Value:
        return self * other

    def __truediv__(self, other: Union[Value, float]):
        raise NotImplemented()

    def __rtruediv__(self, other):  # other / self
        raise NotImplemented()

    def __pow__(self, power: float, modulo=None) -> Value:
        raise NotImplemented()

    def backward(self) -> None:
        self.grad = 1

        for v in find_reversed_topological_order(self):
            v._backwards(v.grad)

    def __repr__(self) -> str:
        return f"Value(data={self.data})"


def find_reversed_topological_order(start: Value) -> list[Value]:
    visited = set()
    order = []
    _build_topological_order(start, visited, order)
    return list(reversed(order))


def _build_topological_order(
    node: Value, visited: set[Value], order: list[Value]
) -> list[Value]:
    if node not in visited:
        visited.add(node)
        for child in node.children:
            _build_topological_order(child, visited, order)

        order.append(node)

    return order
