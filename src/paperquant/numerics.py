"""Small differentiable numeric primitives without file, process or network APIs.

These are shared building blocks; architectures, losses and trading behavior
remain in each independently loaded strategy package.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence


class Scalar:
    def __init__(self, value: float, children: tuple[Scalar, ...] = ()) -> None:
        self.value = float(value)
        self.grad = 0.0
        self.children = children
        self._backward: Callable[[], None] = lambda: None

    @staticmethod
    def lift(value: Scalar | float) -> Scalar:
        return value if isinstance(value, Scalar) else Scalar(value)

    def __add__(self, other: Scalar | float) -> Scalar:
        right = self.lift(other)
        result = Scalar(self.value + right.value, (self, right))

        def backward() -> None:
            self.grad += result.grad
            right.grad += result.grad

        result._backward = backward
        return result

    __radd__ = __add__

    def __neg__(self) -> Scalar:
        return self * -1.0

    def __sub__(self, other: Scalar | float) -> Scalar:
        return self + -self.lift(other)

    def __rsub__(self, other: Scalar | float) -> Scalar:
        return self.lift(other) + -self

    def __mul__(self, other: Scalar | float) -> Scalar:
        right = self.lift(other)
        result = Scalar(self.value * right.value, (self, right))

        def backward() -> None:
            self.grad += right.value * result.grad
            right.grad += self.value * result.grad

        result._backward = backward
        return result

    __rmul__ = __mul__

    def __pow__(self, exponent: float) -> Scalar:
        result = Scalar(self.value**exponent, (self,))
        result._backward = lambda: setattr(
            self, "grad", self.grad + exponent * self.value ** (exponent - 1) * result.grad
        )
        return result

    def __truediv__(self, other: Scalar | float) -> Scalar:
        return self * self.lift(other) ** -1.0

    def __rtruediv__(self, other: Scalar | float) -> Scalar:
        return self.lift(other) * self**-1.0

    def tanh(self) -> Scalar:
        result = Scalar(math.tanh(self.value), (self,))
        result._backward = lambda: setattr(
            self, "grad", self.grad + (1.0 - result.value**2) * result.grad
        )
        return result

    def sigmoid(self) -> Scalar:
        result = Scalar(
            1.0 / (1.0 + math.exp(-self.value))
            if self.value >= 0
            else math.exp(self.value) / (1.0 + math.exp(self.value)),
            (self,),
        )
        result._backward = lambda: setattr(
            self, "grad", self.grad + result.value * (1.0 - result.value) * result.grad
        )
        return result

    def relu(self) -> Scalar:
        result = Scalar(max(0.0, self.value), (self,))
        result._backward = lambda: setattr(self, "grad", self.grad + (self.value > 0) * result.grad)
        return result

    def exp(self) -> Scalar:
        result = Scalar(math.exp(self.value), (self,))
        result._backward = lambda: setattr(self, "grad", self.grad + result.value * result.grad)
        return result

    def log(self) -> Scalar:
        result = Scalar(math.log(self.value), (self,))
        result._backward = lambda: setattr(self, "grad", self.grad + result.grad / self.value)
        return result

    def backward(self) -> None:
        order: list[Scalar] = []
        visited: set[Scalar] = set()
        pending = [(self, False)]
        while pending:
            node, expanded = pending.pop()
            if expanded:
                order.append(node)
            elif node not in visited:
                visited.add(node)
                pending.append((node, True))
                pending.extend((child, False) for child in reversed(node.children))
        for node in order:
            node.grad = 0.0
        self.grad = 1.0
        for node in reversed(order):
            node._backward()


class DenseMLP:
    def __init__(self, widths: Sequence[int], rng: random.Random) -> None:
        self.widths = tuple(widths)
        self.weights = [
            [
                [Scalar(rng.uniform(-1.0, 1.0) / math.sqrt(nin)) for _ in range(nin)]
                for _ in range(nout)
            ]
            for nin, nout in zip(widths, widths[1:], strict=False)
        ]
        self.biases = [[Scalar(0.0) for _ in range(nout)] for nout in widths[1:]]

    def __call__(self, inputs: Sequence[Scalar | float]) -> list[Scalar]:
        values = [Scalar.lift(item) for item in inputs]
        if len(values) != self.widths[0]:
            raise ValueError("MLP input dimension differs from architecture")
        for index, (weights, biases) in enumerate(zip(self.weights, self.biases, strict=True)):
            values = [
                sum((weight * value for weight, value in zip(row, values, strict=True)), bias)
                for row, bias in zip(weights, biases, strict=True)
            ]
            if index != len(self.weights) - 1:
                values = [value.tanh() for value in values]
        return values

    def parameters(self) -> list[Scalar]:
        return [weight for layer in self.weights for row in layer for weight in row] + [
            bias for layer in self.biases for bias in layer
        ]


class Conv1D:
    def __init__(
        self, incoming: int, outgoing: int, kernel: int, rng: random.Random, *, padding: int = 0
    ) -> None:
        self.kernel, self.padding = kernel, padding
        self.weights = [
            [
                [Scalar(rng.uniform(-1, 1) / math.sqrt(incoming * kernel)) for _ in range(kernel)]
                for _ in range(incoming)
            ]
            for _ in range(outgoing)
        ]
        self.biases = [Scalar(0.0) for _ in range(outgoing)]

    def __call__(self, channels: Sequence[Sequence[Scalar | float]]) -> list[list[Scalar]]:
        if len(channels) != len(self.weights[0]) or not channels:
            raise ValueError("Convolution input channels differ")
        length = len(channels[0])
        if any(len(channel) != length for channel in channels):
            raise ValueError("Convolution channels have different lengths")
        padded = [
            [Scalar(0.0)] * self.padding
            + [Scalar.lift(v) for v in channel]
            + [Scalar(0.0)] * self.padding
            for channel in channels
        ]
        width = length + 2 * self.padding - self.kernel + 1
        if width <= 0:
            raise ValueError("Convolution window is shorter than its kernel")
        return [
            [
                sum(
                    (
                        weight * padded[i][t + k]
                        for i, row in enumerate(weights)
                        for k, weight in enumerate(row)
                    ),
                    bias,
                )
                for t in range(width)
            ]
            for weights, bias in zip(self.weights, self.biases, strict=True)
        ]

    def parameters(self) -> list[Scalar]:
        return [weight for group in self.weights for row in group for weight in row] + self.biases


class LSTMCell:
    def __init__(self, incoming: int, hidden: int, rng: random.Random) -> None:
        self.hidden = hidden
        self.gates = [DenseMLP((incoming + hidden, hidden), rng) for _ in range(4)]

    def __call__(
        self, inputs: Sequence[Scalar | float], hidden: Sequence[Scalar], cell: Sequence[Scalar]
    ) -> tuple[list[Scalar], list[Scalar]]:
        combined = list(hidden) + list(inputs)
        candidate = [v.tanh() for v in self.gates[0](combined)]
        update, forget, output = [[v.sigmoid() for v in gate(combined)] for gate in self.gates[1:]]
        next_cell = [
            u * c + f * old for u, c, f, old in zip(update, candidate, forget, cell, strict=True)
        ]
        next_hidden = [o * c.tanh() for o, c in zip(output, next_cell, strict=True)]
        return next_hidden, next_cell

    def parameters(self) -> list[Scalar]:
        return [value for gate in self.gates for value in gate.parameters()]


def gradient_step(parameters: Sequence[Scalar], learning_rate: float, *, clip: float = 5.0) -> None:
    if learning_rate < 0 or clip <= 0:
        raise ValueError("Invalid optimizer settings")
    for parameter in parameters:
        if not math.isfinite(parameter.grad):
            raise ValueError("Training produced a nonfinite gradient")
        parameter.value -= learning_rate * max(-clip, min(clip, parameter.grad))
        if not math.isfinite(parameter.value):
            raise ValueError("Training produced a nonfinite parameter")
