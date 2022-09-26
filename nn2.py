from random import choices
from warnings import warn
import numpy as np
from typing import Callable
from numpy import ndarray
from sympy import Symbol, lambdify


class Layer:
    neurons: int

    def feed_forward(self, input: ndarray) -> tuple[ndarray, ndarray]:
        raise NotImplementedError("abstract method")

    def calc_derivative(self, output: ndarray) -> ndarray:
        raise NotImplementedError("abstract method")

    def calc_delta(self, prev_delta) -> ndarray:
        raise NotImplementedError("abstract method")

    def get_wshape(self) -> tuple:
        raise NotImplementedError("abstract method")

    def get_bshape(self) -> tuple:
        raise NotImplementedError("abstract method")

    def update_weights(self, batch_size, learning_rate, dW, dB):
        raise NotImplementedError("abstract method")


class DenseLayer(Layer):
    w: ndarray
    b: ndarray
    act: Callable[[ndarray], ndarray]
    der: Callable[[ndarray], ndarray]

    def __repr__(self) -> str:
        return f"Dense{{{self.neurons}}}"

    def __init__(self, w: ndarray, b: ndarray, act: Callable[[ndarray], ndarray], der: Callable[[ndarray], ndarray]):
        self.w = w
        self.b = b
        self.act = act
        self.der = der
        self.neurons = w.shape[1]

    def feed_forward(self, input: ndarray) -> tuple[ndarray, ndarray]:
        # multiply by weights & add a bias
        z = self.w.T @ input + self.b
        # activate
        a = self.act(z)
        return z, a

    def calc_derivative(self, input: ndarray) -> ndarray:
        return self.der(input)

    def calc_delta(self, prev_delta) -> ndarray:
        return self.w @ prev_delta

    def get_wshape(self) -> tuple:
        return self.w.shape

    def get_bshape(self) -> tuple:
        return self.b.shape

    def update_weights(self, batch_size, learning_rate, dW, dB):
        self.w -= dW * learning_rate / batch_size
        self.b -= dB * learning_rate / batch_size


class LayerModel:
    act: Callable[[ndarray], ndarray]
    der: Callable[[ndarray], ndarray] | None = None
    neurons: int
    _type: str

    def activation(self, act: Callable[[ndarray], ndarray]):
        self.act = act

    def activation_derivative(self, der: Callable[[ndarray], ndarray]):
        self.der = der

    def __init__(self, neurons: int, type: str = "dense") -> None:
        self.neurons = neurons
        self.act = lambda x: x
        self._type = type

    def build(self, input_shape: int) -> DenseLayer:
        der: Callable[[ndarray], ndarray]
        if not isinstance(self.der, type(None)):
            der = self.der
        else:
            x = Symbol('x')
            y = self.act(x)  # type: ignore
            yp = y.diff(x)  # type: ignore
            der = lambdify(x, yp, 'numpy')

        match self._type:
            case "dense":
                w = np.random.randn(input_shape, self.neurons)
                b = np.random.randn(self.neurons, 1)
                return DenseLayer(w, b, self.act, der)
            case _:
                raise NotImplementedError(
                    "Other types are not implemented yet")


class NeuNet:
    layers: list[Layer]
    input_shape: int

    def check_inputs(self, inputs: list[ndarray]):
        for input in inputs:
            if input.shape[0] != self.input_shape:
                raise ValueError("Invalid inputs provided")

    def check_net(self):
        if len(self.layers) == 0:
            raise ValueError("Invalid output layer")

    def _back_prop(self, x: ndarray, y: ndarray) -> tuple[list[ndarray], list[ndarray]]:
        dBns = [np.zeros(0) for _ in self.layers]
        dWns = [np.zeros(0) for _ in self.layers]

        Zn = []
        An = [x.reshape((*x.shape, 1))]
        y = y.reshape((*y.shape, 1))

        for layer in self.layers:
            z, a = layer.feed_forward(An[-1])
            Zn.append(z)
            An.append(a)

        H = len(self.layers)-1
        derivative = self.layers[H].calc_derivative(Zn[H])
        delta = (An[H+1]-y) * derivative  # type: ignore
        dBns[H] = delta  # type: ignore
        dWns[H] = An[H] @ delta.T  # type: ignore
        for i in range(H-1, -1, -1):
            derivative = self.layers[i].calc_derivative(Zn[i])
            delta = self.layers[i+1].calc_delta(delta) * derivative
            dBns[i] = delta  # type: ignore
            dWns[i] = An[i] @ delta.T  # type: ignore

        return dBns, dWns

    def _get_batch(self, xn: list[ndarray] | ndarray, yn: list[ndarray] | ndarray, batch_size: int):
        if len(xn) != len(yn):
            raise ValueError("Mismatched data sizes")
        if len(xn) < batch_size:
            warn("Batch size is greater than data size")
        return choices(list(zip(xn, yn)), k=batch_size)

    def fit(self, x_train: list[ndarray] | ndarray, y_train: list[ndarray] | ndarray, learning_rate: float = 5.0, passes=10, batch_size=6):
        self.check_net()
        if isinstance(x_train, list):
            self.check_inputs(x_train)
        else:
            self.check_inputs(list(x_train))

        for _ in range(passes):
            dBn = [np.zeros(layer.get_bshape()) for layer in self.layers]
            dWn = [np.zeros(layer.get_wshape()) for layer in self.layers]

            for x, y in self._get_batch(x_train, y_train, batch_size):
                dBns, dWns = self._back_prop(x, y)
                for dB, dBs, dW, dWs in zip(dBn, dBns, dWn, dWns):
                    dB += dBs
                    dW += dWs

            for layer, dB, dW in zip(self.layers, dBn, dWn):
                layer.update_weights(batch_size, learning_rate, dW, dB)

    def predict(self, inputs: ndarray) -> ndarray:
        result = [0, inputs.reshape((*inputs.shape, 1))]
        self.check_net()
        self.check_inputs([inputs])

        for layer in self.layers:
            result = layer.feed_forward(result[1])
        return result[1]

    def __init__(self, input_shape: int) -> None:
        self.layers = []
        self.input_shape = input_shape

    def add_layer(self, layer: LayerModel | Layer) -> None:
        prev_shape = self.input_shape if len(
            self.layers) == 0 else self.layers[-1].neurons
        if isinstance(layer, LayerModel):
            self.layers.append(layer.build(prev_shape))
        elif isinstance(layer, Layer):
            if layer.get_wshape()[0] != prev_shape:
                raise ValueError("Invalid layer provided")
            self.layers.append(layer)
        else:
            raise ValueError("Unsupported layer type")


def sigmoid(x): return 1./(1.+np.e**(-x))


def tanh(x): return (np.e**x-np.e**(-x))/(np.e**x+np.e**(-x))
