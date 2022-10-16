import torch
from graphviz import Digraph

from mini_auto_grad.engine import Value


def trace(root: Value) -> tuple[set[Value], set[Value]]:
    nodes = set()
    edges = set()

    def _build(v: Value):
        if v not in nodes:
            nodes.add(v)
            for child in v.children:
                edges.add((child, v))
                _build(child)

    _build(root)
    return nodes, edges


def draw_graph(root: Value, format="svg", rankdir="LR"):
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={"rankdir": rankdir})

    for n in nodes:
        dot.node(
            name=str(id(n)),
            label=f"data={n.data:.4f} | grad={n.grad:.4f}",
            # label=f"data={n.data:.4f} | ",
            shape="record",
        )
        if n._op:
            dot.node(name=str(id(n)) + n._op, label=n._op)
            dot.edge(str(id(n)) + n._op, str(id(n)))

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot


if __name__ == "__main__":
    # brew install graphviz might be needed
    a = Value(1)
    b = Value(2)
    c = Value(3)
    d = Value(4)

    y = a + b + c + d
    draw_graph(y).render(filename="g1.dot")
