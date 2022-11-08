# Mini auto grad
The goal of this project is to create a mini auto gradient engine.
We will do this in two steps:
1. Create the `Value` data structure that keeps track of the computational graph when you use the default math operator `+`, `*`, `**`, etc.
2. Implement the [backpropagation](https://en.wikipedia.org/wiki/Backpropagation) algorithm using the computational graph.


When you are done, you can create a computational graph and calculates its gradients using the following API:


```python
x_1 = Value(3)
x_2 = Value(4)
x_3 = x_1 * x_2
x_4 = Value(5)
x_5 = x_4 + x_3
x_6 = Value(2)
l = x_5 * x_6

l.backwards()
```

![example computationgraph](images/computation_graph.png)
A Visualization of the above computational graph and its gradients.

## Derivative rules overview
You do not need to remember much from calculus classes. As long as you can remember the following derivative rules you should be fine.

| Name                         |    Function     |                                                                         Derivative                                                                          |
|------------------------------|:---------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------:|
| Definition                   |        -        |                                                             $\frac{\partial L}{\partial L} = 1$                                                             |
| (Unrelated) constant rule    |     $L = c$     |                                                             $\frac{\partial L}{\partial x} = 0$                                                             |
| Sum rule                     | $L = x_1 + x_2$ |                                        $\frac{\partial L}{\partial x_1} = 1$, $\frac{\partial L}{\partial x_2} = 1$                                         |
| Multiplication rule          | $L = x_1 * x_2$ |                                      $\frac{\partial L}{\partial x_1} = x_2$, $\frac{\partial L}{\partial x_2} = x_1$                                       |
| Constant multiplication rule |    $L = c*x$    |                                                             $\frac{\partial L}{\partial x} = c$                                                             |
| Power rule                   |  $L = c * x^n$  |                                                      $\frac{\partial L}{\partial x} = c * n * x^{n-1}$                                                      |
| Chain rule                   |        -        | $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial x} \frac{\partial y}{\partial y} = \frac{\partial L}{\partial y} \frac{\partial x}{\partial y}$ |
| tanh                         |     tanh(x)     |    $\frac{\partial L}{\partial x} = 1 - (tanh(x))^2$                                                                                                                                                          |



## Exercises
First install the project using:
```bash
poetry run install
```

You might need to tell poetry where to find a python 3.9 interpreter using:
```bash
poetry env use /path/to/python
```

### Exercise 1: Implement the computational graph.
In this exercise we are going to implement a data structure that keeps track of the computational graph.
This data structure will be implemented in the `Value` class in `src/mini_auto_grad/engine.py`.
To implement this functionality you have to do the following:
1. Implement the arithmetic logic by using the `__add__`, `____radd__`, `__pow__`,  `__mul__`, etc such that the returned value has the correct `data` attribute.
2. Ensure that the new `Value` instance that is returned by the arithmetic method keeps track of the children that created it. E.g. `x = a + b` then `a` and `b` will be children of `x`.


You can verify that you have implemented everything correctly using:
```bash
poetry run pytest -m ex1
```

### Exercise 2: Implement back propagation
In this exercise we are going to implement the back propagation algorithm in the `backwards` method of the `Value` class.
This algorithm does the following:
1. Call the `backwards` method on the loss leaf node.
2. Set the gradient of this loss leaf node to `1` since $\frac{\partial L}{\partial L} = 1$.
3. Find the revered topological order of the computation graph using dfs.
4. Propagate the gradient backwards by calling the `_backwards(grad)` function.

To implement this algorithm you need to do the following:
1. Make sure that arithmetic functions (`__add__`, `____radd__`, `__pow__`,  `__mul__`, etc) pass the correct `_backwards` function to their parent node.
2. Finish the `find_reversed_topological_order` function in `src/mini_auto_grad/engine.py`.


You can verify if you have implemented everything correctly using:
```bash
poetry run pytest -m ex2
```

### Optional Exercise 3: Create MLP
We have a working auto grad engine, lets see if we can use it to build a small MLP that can solve the XOR problem.
We have set up this MLP in learning process in `tests/test_nn.py`. 

To implement this functionality you have to do the following in `src/test_nn.py`:
1. Finish the implementation of `Neuron`. This should be a function $f(x) = b + \sum_{i=0}^N x_i * w_i$ whereby $b$ and $w_i$ are instances of `Value`.
2. Finish the implementation of `Layer`. This function that takes $n$ inputs and uses $m$ `Neuron` functions to map it to $m$ output features.

You can verify if you have implemented everything correctly using:
```bash
poetry run pytest -m ex3
```
