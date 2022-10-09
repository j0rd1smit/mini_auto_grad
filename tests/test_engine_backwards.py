import pytest
import torch

from mini_auto_grad.engine import Value


@pytest.mark.parametrize("data", [-1.0, 0, 1.0, 10], ids=lambda d: f"data={d}")
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
def test_example(a: float, b: float) -> None:
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
def test_example1(a: float, b: float) -> None:
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
        (-3, 9, 1),
        (0, 1, 8),
    ],
)
def test_example_2(a: float, b: float, c: float) -> None:
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


@pytest.mark.parametrize(
    "a,b,c",
    [(4, -1, 1), (2, 2, 2), (-3, 9, 1), (0, 1, 8)],
)
def test_example_3(a: float, b: float, c: float) -> None:
    a = Value(a)
    b = Value(b)
    c = Value(c)

    d = a + b
    e = d * c

    e.backward()

    # d_e / d_e = 1
    assert e.grad == 1
    # d_d / d_e = (d_d / d_e) * (d_e * d_e)
    assert d.grad == c.data * 1
    # d_c / d_e = (d_c / d_e) * (d_e * d_e)
    assert c.grad == d.data * 1
    # d_b / d_e = (d_b / d_d) * (d_d * d_e)
    assert b.grad == d.grad
    # d_a / d_e = (d_a / d_d) * (d_d * d_e)
    assert a.grad == d.grad


def test_sanity_check():

    x = Value(-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xmg, ymg = x, y

    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xpt, ypt = x, y

    # forward pass went well
    assert ymg.data == ypt.data.item()
    # backward pass went well
    assert xmg.grad == xpt.grad.item()
