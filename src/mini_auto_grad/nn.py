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
        self.weights = []  # TODO ex3
        self.bias = ...  # TODO ex3
        self.non_linear = non_linear

    def parameters(self) -> list[Value]:
        return ...  # TODO ex3

    def __call__(self, x: list[Union[Value, float]]) -> Value:
        """This is a function f(x) = b + \sum_{i=0}^N x_i * w_i"""
        activation = ...  # TODO ex3

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

        self.neurons = [...]  # TODO ex3

    def __call__(self, x) -> list[Value]:
        """This function that takes $n$ inputs and uses $m$ `Neuron` functions to map it to $m$ output features."""
        return ...  # TODO ex3

    def parameters(self) -> list[Value]:
        return [...]  # TODO ex3

    def __repr__(self):
        return f"Layer({self.n_features_in}, {self.n_features_out}, {self.non_linear})"


class MLP(Module):
    def __init__(self, sizes: list[int]) -> None:
        self.sizes = sizes
        self.layers = _create_layers(sizes)

    def parameters(self):
        return [...]  # TODO ex3

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
