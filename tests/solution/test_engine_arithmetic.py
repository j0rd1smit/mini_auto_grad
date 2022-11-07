import math
from typing import Union

import pytest

from mini_auto_grad.solution.engine import Value


@pytest.mark.parametrize(
    "left,right,expected_output_data",
    [
        (Value(1), 1, 2),
        (1, Value(2), 3),
        (Value(3), Value(1), 4),
        (Value(3), -1, 2),
        (-1, Value(3), 2),
    ],
)
@pytest.mark.solution()
def test_addition(
    left: Union[float, Value], right: Union[float, Value], expected_output_data: float
) -> None:
    output = left + right

    assert output.data == expected_output_data


@pytest.mark.solution()
def test_addition_graph() -> None:
    left = Value(1)
    right = Value(1)
    output = left + right

    assert _is_connected(output, left)
    assert _is_connected(output, right)


def _is_connected(value: Value, other: Value) -> bool:
    visited = set()
    queue = [value]
    while len(queue) > 0:
        node = queue.pop()
        visited.add(value)

        if node == other:
            return True

        for child in node.children:
            if child not in visited:
                queue.append(child)

    return False


@pytest.mark.parametrize(
    "left,right,expected_output_data",
    [
        (Value(1), 1, 0),
        (1, Value(2), -1),
        (Value(3), Value(1), 2),
        (Value(3), -1, 4),
        (-1, Value(3), -4),
    ],
)
@pytest.mark.solution()
def test_subtraction(
    left: Union[float, Value], right: Union[float, Value], expected_output_data: float
) -> None:
    output = left - right

    assert output.data == expected_output_data


@pytest.mark.solution()
def test_subtraction_graph() -> None:
    left = Value(1)
    right = Value(1)
    output = left - right

    assert _is_connected(output, left)
    assert _is_connected(output, right)


@pytest.mark.parametrize(
    "left,right,expected_output_data",
    [
        (Value(1), 1, 1),
        (1, Value(2), 2),
        (Value(3), Value(2), 6),
        (Value(3), -1, -3),
        (-3, Value(3), -9),
    ],
)
@pytest.mark.solution()
def test_multiplication(
    left: Union[float, Value], right: Union[float, Value], expected_output_data: float
) -> None:
    output = left * right

    assert output.data == expected_output_data


@pytest.mark.solution()
def test_multiplication_graph() -> None:
    left = Value(1)
    right = Value(1)
    output = left * right

    assert _is_connected(output, left)
    assert _is_connected(output, right)


@pytest.mark.parametrize(
    "left,right,expected_output_data",
    [
        (Value(1), 1, 1),
        (1, Value(2), 0.5),
        (Value(3), Value(2), 3 / 2),
        (Value(3), -1, -3),
        (-21, Value(3), -7),
    ],
)
@pytest.mark.solution()
def test_division(
    left: Union[float, Value], right: Union[float, Value], expected_output_data: float
) -> None:
    output = left / right

    assert output.data == expected_output_data


@pytest.mark.solution()
def test_division_graph() -> None:
    left = Value(1.0)
    right = Value(1.0)
    output = left / right

    assert _is_connected(output, left)
    assert _is_connected(output, right)


@pytest.mark.parametrize(
    "left,right,expected_output_data",
    [
        (Value(1), 1, 1),
        (Value(2), 2, 4),
        (Value(3), 5, 243),
        (Value(3), -1, 1 / 3),
        (Value(-3), 2, 9),
    ],
)
@pytest.mark.solution()
def test_power(
    left: Union[float, Value], right: Union[float, Value], expected_output_data: float
) -> None:
    output = left**right

    assert output.data == expected_output_data


@pytest.mark.solution()
def test_power_graph() -> None:
    left = Value(3)
    output = left**2

    assert _is_connected(output, left)


@pytest.mark.parametrize(
    "value,expected_output_data",
    [
        (0, 0),
        (-100, -1),
        (100, 1),
        (-0.5, math.tanh(-0.5)),
        (0.5, math.tanh(0.5)),
    ],
)
@pytest.mark.solution()
def test_tanh(value: float, expected_output_data: float) -> None:
    value = Value(value)
    assert value.tanh().data == expected_output_data


@pytest.mark.solution()
def test_tanh_graph() -> None:
    value = Value(2)
    output = value.tanh()
    assert _is_connected(output, value)
