import math
import numpy as np
import random


class Value:
    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data
        self.grad = 0.00  # gradient of this Value, used in backpropagation (how much this Value contributes to the final output)
        self._backward = (
            lambda: None
        )  # this function will change, depending on the operation
        self._prev = set(_children)  # a Value can be the result of multiple parents
        self._op = _op  # the operation that produced this Value, used in visual representations
        self.label = label  # a label for the node, used in visual representations

    def __repr__(self):
        return (
            f"Value(data={self.data})"  # printing the Value object will show the data
        )

    # we need to define all the possible operations for our Value class

    def __add__(self, other):
        # this is done in order to add constants to Values that are not Values themselves
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        # += grad because this node can be used multiple times during backpropagation
        def _backward():
            self.grad += 1.0 * out.grad  # (see sum example backpropagation below)
            other.grad += 1.0 * out.grad  # (see sum example backpropagation below)

        out._backward = _backward
        return out

    def __neg__(self):  # -self
        return self * -1

    def __radd__(self, other):  # other + self
        return self + other

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def __rmul__(self, other):  # other * self
        return self * other

    def __truediv__(self, other):  # self / other
        return self * other**-1

    def __rtruediv__(self, other):  # other / self
        return other * self**-1

    def __mul__(self, other):
        # this is done in order to multiply constants to Values that are not Values themselves
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # local derivative (out.grad) multiplied by other.data, as per the chain rule
            self.grad += (
                other.data * out.grad
            )  # (see product example backpropagation below)
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    # function called when a Value is raised to a power and other is that power
    def __pow__(self, other):
        assert isinstance(
            other, (int, float)
        ), "Power must be int or float, supporting only this for now"
        out = Value(self.data**other, (self,), f"pow_{other}")

        def _backward():
            self.grad += (
                other * self.data ** (other - 1) * out.grad
            )  # power rule: d/dx x^n = n*x^(n-1), and then chain rule (* out.grad)

        out._backward = _backward
        return out

    def __truediv__(self, other):
        return self * other**-1

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), "exp")  # out.data used below

        def _backward():
            self.grad += out.data * out.grad  # d/dx exp(x) = exp(x)

        out._backward = _backward
        return out

    # used inthe neuron example, as the activation function
    def tanh(self):
        n = self.data
        t = (math.exp(2 * n) - 1) / (math.exp(2 * n) + 1)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out
    
    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def backward(self):
        self.grad = 1.0
        visited = set()
        topo_order = []

        def dfs(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    dfs(child)
                topo_order.append(v)

        dfs(self)
        for v in reversed(topo_order):
            v._backward()


class Neuron:
    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]  # weights
        self.b = Value(random.uniform(-1, 1))  # bias
        self.nonlin = nonlin  # whether the neuron has an activation function

    def __call__(self, x):
        act = (
            sum(wi * xi for wi, xi in zip(self.w, x)) + self.b
        )  # weighted sum of inputs
        out = act.tanh()  # activation function
        return out

    def parameters(self):
        return self.w + [self.b]
    
    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"


class Layer:
    # nin is the number of inputs, nouts represents the number of neurons in the layer
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]  # multiple neurons in a layer

    def __call__(self, x):
        outs = [neuron(x) for neuron in self.neurons]  # list of outputs for each neuron
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        params = []
        for neuron in self.neurons:
            ps = neuron.parameters()
            params.extend(ps)
        return params
    
    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP:
    # now we do not take a single value for nouts, but a list of it, to represent the number of neurons in each layer
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(sz) - 1)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0
    
    def __repr__(self):
        layers_str = '\n'.join(str(layer) for layer in self.layers)
        return f"MLP of:\n{layers_str}"
