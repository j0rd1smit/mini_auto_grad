import abc
import random
from typing import Union

from mini_auto_grad.engine import Value


class Module(abc.ABC):
    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = 0.0

    @abc.abstractmethod
    def parameters(self) -> list[Value]:
        pass


class Neuron(Module):
    def __init__(self, n_features: int, non_linear: bool = True) -> None:
        self.weights = [Value(random.uniform(-1, 1)) for _ in range(n_features)]
        self.bias = Value(0)
        self.non_linear = non_linear

    def parameters(self) -> list[Value]:
        return self.weights + [self.bias]

    def __call__(self, x: list[Union[Value, float]]) -> Value:
        activation = sum(w_i * x_i for w_i, x_i in zip(self.weights, x)) + self.bias

        if self.non_linear:
            return activation.tanh()

        return activation

    def __repr__(self) -> str:
        return f"Neuron(n_features={len(self.weights)}, non_linear={self.non_linear})"


class Layer(Module):
    def __init__(
        self, n_features_in: int, n_features_out: int, non_linear: bool
    ) -> None:
        self.n_features_in = n_features_in
        self.n_features_out = n_features_out
        self.non_linear = non_linear

        self.neurons = [
            Neuron(n_features_in, non_linear) for _ in range(n_features_out)
        ]

    def __call__(self, x) -> list[Value]:
        return [n(x) for n in self.neurons]

    def parameters(self) -> list[Value]:
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer({self.n_features_in}, {self.n_features_out}, {self.non_linear})"


class MLP(Module):
    def __init__(self, sizes: list[int]) -> None:
        self.sizes = sizes
        self.layers = _create_layers(sizes)

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def __repr__(self):
        return f"MLP({self.sizes})"


def _create_layers(sizes: list[int]) -> list[Layer]:
    layers = []

    for i in range(len(sizes) - 1):
        is_not_final_layer = i != len(sizes) - 1
        layers.append(Layer(sizes[i], sizes[i + 1], non_linear=is_not_final_layer))

    return layers
