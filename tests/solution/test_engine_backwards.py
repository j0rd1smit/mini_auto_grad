import math

import pytest

from mini_auto_grad.solution.engine import Value


@pytest.mark.parametrize("data", [-1.0, 0, 1.0, 10], ids=lambda d: f"data={d}")
@pytest.mark.solution()
def test_gradient_with_respect_to_self_is_one(data: float) -> None:
    value = Value(data)
    assert value.grad == 0

    value.backward()

    assert value.grad == 1


@pytest.mark.parametrize(
    "a,b",
    [
        (4, -1),
        (2, 2),
        (-3, 9),
        (0, 1),
    ],
)
@pytest.mark.solution()
def test_multiplication(a: float, b: float) -> None:
    a = Value(a)
    b = Value(b)
    c = a * b

    c.backward()

    # d_a / d_c = (d_a / d_c) * (d_c / d_c)
    # d_a / d_c = b (scalar rule)
    # d_c / d_c = 1
    assert a.grad == b.data * 1
    # d_b / d_c = (d_b / d_c) * (d_c / d_c)
    # d_a / d_c = a (scalar rule)
    # d_c / d_c = 1
    assert b.grad == a.data * 1


@pytest.mark.parametrize(
    "a,b",
    [(4, -1), (2, 2), (-3, 9), (0, 1)],
)
def test_addition(a: float, b: float) -> None:
    a = Value(a)
    b = Value(b)
    c = a + b

    c.backward()

    # d_a / d_c = (d_a / d_c) * (d_c / d_c)
    # d_a / d_c = 1 (constant rule)
    # d_c / d_c = 1
    assert a.grad == 1
    # d_b / d_c = (d_b / d_c) * (d_c / d_c)
    # d_a / d_c = 1 (constant rule)
    # d_c / d_c = 1
    assert b.grad == 1


@pytest.mark.parametrize(
    "a,b,c",
    [
        (4, -1, 1),
        (2, 2, 2),
        (-3, 9, 2),
        (0, 1, 8),
    ],
)
@pytest.mark.solution()
def test_deep_multiplication(a: float, b: float, c: float) -> None:
    a = Value(a)
    b = Value(b)
    c = Value(c)

    d = a * b
    e = d * c

    e.backward()

    # d_e / d_e = 1
    assert e.grad == 1
    # d_d / d_e = (d_d / d_e) * (d_e * d_e)
    assert d.grad == c.data * 1
    # d_c / d_e = (d_c / d_e) * (d_e * d_e)
    assert c.grad == d.data * 1
    # d_b / d_e = (d_b / d_d) * (d_d * d_e)
    assert b.grad == a.data * d.grad
    # d_a / d_e = (d_a / d_d) * (d_d * d_e)
    assert a.grad == b.data * d.grad


@pytest.mark.solution()
@pytest.mark.parametrize("n_additions", [1, 3, 5, 10], ids=lambda d: f"n_additions={d}")
def test_multiplication_in_the_form_of_addition(n_additions: int):
    value = Value(1)

    output = Value(1)
    for _ in range(n_additions):
        output += value

    output.backward()
    assert value.grad == n_additions


@pytest.mark.parametrize("raw_value", [-1, 0, 1], ids=lambda d: f"raw_value={d}")
@pytest.mark.solution()
def test_tanh_backwards(raw_value: float):
    value = Value(raw_value)

    output = value.tanh()
    output.backward()

    assert value.grad == 1 - math.tanh(raw_value) ** 2
