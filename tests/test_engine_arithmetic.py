import math
from typing import Union

import pytest

from mini_auto_grad.engine import Value


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
def test_addition(
    left: Union[float, Value], right: Union[float, Value], expected_output_data: float
) -> None:
    output = left + right

    assert output.data == expected_output_data


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
def test_subtraction(
    left: Union[float, Value], right: Union[float, Value], expected_output_data: float
) -> None:
    output = left - right

    assert output.data == expected_output_data


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
def test_multiplication(
    left: Union[float, Value], right: Union[float, Value], expected_output_data: float
) -> None:
    output = left * right

    assert output.data == expected_output_data


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
def test_division(
    left: Union[float, Value], right: Union[float, Value], expected_output_data: float
) -> None:
    output = left / right

    assert output.data == expected_output_data


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
def test_power(
    left: Union[float, Value], right: Union[float, Value], expected_output_data: float
) -> None:
    output = left**right

    assert output.data == expected_output_data


@pytest.mark.parametrize(
    "value,expected_output_data",
    [
        (0, 0),
        (1e-9, 1e-9),
        (2, 2),
        (-1e-9, 0),
        (-1, 0),
    ],
)
def test_relu(value: float, expected_output_data: float) -> None:
    assert Value(value).relu().data == expected_output_data


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
def test_tanh(value: float, expected_output_data: float) -> None:
    assert Value(value).tanh().data == expected_output_data
