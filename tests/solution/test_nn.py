import pytest

from mini_auto_grad.solution.engine import Value
from mini_auto_grad.solution.nn import MLP


@pytest.mark.solution()
def test_mlp_can_learn_xor_problem() -> None:
    x = [
        [0, 1],
        [1, 1],
        [0, 0],
        [1, 0],
    ]
    y = [1, 0, 1, 0]

    mlp = MLP([2, 4, 4, 1])

    for _ in range(50):
        loss = _forward(mlp, x, y)

        mlp.zero_grad()
        loss.backward()

        lr = 0.1
        for p in mlp.parameters():
            p.data += -lr * p.grad

    loss = _forward(mlp, x, y)
    assert loss.data <= 0.05


def _forward(mlp: MLP, x: list[list[float]], y: list[float]) -> Value:
    y_pred = [mlp(x_i)[0] for x_i in x]
    return sum((y_i - y_pred_i) ** 2 for y_i, y_pred_i in zip(y, y_pred)) / len(y)
