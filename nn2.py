from random import choices
from warnings import warn
import numpy as np
from typing import Callable
from numpy import array, ndarray
from sympy import Predicate, Symbol, lambdify
from tqdm import trange


class Layer:
    neurons: int

    def feed_forward(self, input: ndarray) -> tuple[ndarray, ndarray]:
        "forward pass"
        raise NotImplementedError("abstract method")

    def calc_derivative(self, input: ndarray) -> ndarray:
        "calculate derivative value with given input"
        raise NotImplementedError("abstract method")

    def calc_delta(self, prev_delta) -> ndarray:
        "calculate delta from previous delta and this layer data"
        raise NotImplementedError("abstract method")

    def get_wshape(self) -> tuple:
        "return shape of weight matrix"
        raise NotImplementedError("abstract method")

    def get_bshape(self) -> tuple:
        "return biases shape"
        raise NotImplementedError("abstract method")

    def update_weights(self, batch_size, learning_rate, dW, dB):
        "update weights based on accumulated deltas from batch, batch_size and learning rate"
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
    # activation function
    act: Callable[[ndarray], ndarray]
    # derivative of activation function
    der: Callable[[ndarray], ndarray] | None = None
    # number of neurons
    neurons: int
    # type of layer (for now only dense is supported)
    _type: str

    def activation(self, act: Callable[[ndarray], ndarray]):
        "change activation function for layer"
        self.act = act

    def activation_derivative(self, der: Callable[[ndarray], ndarray]):
        """change derivative of activation function for layer
        note: changing derivative does not affect activation function itself"""
        self.der = der

    def __init__(self, neurons: int, type: str = "dense") -> None:
        self.neurons = neurons
        self.act = lambda x: x
        self._type = type

    def build(self, input_shape: int) -> DenseLayer:
        "generate layer from it's model"
        der: Callable[[ndarray], ndarray]
        if not isinstance(self.der, type(None)):
            der = self.der
        else:
            # symbolicly calculate derivative if it's not present explicitly
            x = Symbol('x')
            y = self.act(x)  # type: ignore
            yp = y.diff(x)  # type: ignore
            der = lambdify(x, yp, 'numpy')

        match self._type:
            case "dense":
                # randomly initialize weights and biases
                w = np.random.randn(input_shape, self.neurons)
                b = np.random.randn(self.neurons, 1)
                return DenseLayer(w, b, self.act, der)
            case _:
                raise NotImplementedError(
                    "Other types are not implemented yet")


class NeuNet:
    # list of layers
    layers: list[Layer]
    # shape of inputs to neural network
    input_shape: int

    def check_inputs(self, inputs: list[ndarray]):
        "check list of inputs against this network input_shape"
        for input in inputs:
            if input.shape[0] != self.input_shape:
                raise ValueError("Invalid inputs provided")

    def check_net(self):
        "check wheither this network has any layer"
        if len(self.layers) == 0:
            raise ValueError("Invalid output layer")

    def _back_prop(self, x: ndarray, y: ndarray) -> tuple[list[ndarray], list[ndarray]]:
        "backward propagation"
        # init lists of deltas for weights and biases for each layer
        dBns = [np.zeros(0) for _ in self.layers]
        dWns = [np.zeros(0) for _ in self.layers]

        # list of outputs of layers before applying activation function
        Zn = []
        # list of outputs of layers after aaf
        An = [x.reshape((*x.shape, 1))]
        y = y.reshape((*y.shape, 1))

        # forward pass saving all layer's outputs
        for layer in self.layers:
            z, a = layer.feed_forward(An[-1])
            Zn.append(z)
            An.append(a)

        H = len(self.layers)-1
        # calculate derivative based on last layer's output
        derivative = self.layers[H].calc_derivative(Zn[H])
        # calculate delta of last layer
        delta = (An[H+1]-y) * derivative  # type: ignore
        # save delta of last layer
        dBns[H] = delta  # type: ignore
        dWns[H] = An[H] @ delta.T  # type: ignore
        # calculate other layer's deltas and save'em
        for i in range(H-1, -1, -1):
            derivative = self.layers[i].calc_derivative(Zn[i])
            delta = self.layers[i+1].calc_delta(delta) * derivative
            dBns[i] = delta  # type: ignore
            dWns[i] = An[i] @ delta.T  # type: ignore

        return dBns, dWns

    def _get_batch(self, xn: list[ndarray] | ndarray, yn: list[ndarray] | ndarray, batch_size: int) \
            -> list[tuple[list[int], ndarray, ndarray]]:
        "get random batch for given inputs `xn` and expected outputs `yn` of size `batch_size`"
        if len(xn) != len(yn):
            raise ValueError("Mismatched data sizes")
        if len(xn) < batch_size:
            warn("Batch size is greater than data size")
        idxs = list(range(len(xn)))
        return choices(list(zip(idxs, xn, yn)), k=batch_size)   # type: ignore

    def fit(self,
            x_train: list[ndarray] | ndarray,
            y_train: list[ndarray] | ndarray,
            learning_rate: float = 5.0,
            passes=10, batch_size=6,
            cross_validation: Callable[[ndarray, ndarray], float] | None = None):
        """train network with back propagation with provided input data `x_train`, 
        expected output `y_train`, `learning_rate`, 
        number of `passes`, `batch_size` and optional `cross_validation` function"""
        self.check_net()
        if isinstance(x_train, list):
            self.check_inputs(x_train)
        else:
            self.check_inputs(list(x_train))

        # tqdm's iterator with progress bar
        with trange(passes) as pbar:
            pbar.set_description("Training")
            for _ in pbar:
                # init deltas for current batch
                dBn = [np.zeros(layer.get_bshape()) for layer in self.layers]
                dWn = [np.zeros(layer.get_wshape()) for layer in self.layers]

                excluded = []
                # run backward propagation on generated batch
                for i, x, y in self._get_batch(x_train, y_train, batch_size):
                    excluded.append(i)
                    dBns, dWns = self._back_prop(x, y)
                    # accumulate deltas for all layers
                    for dB, dBs, dW, dWs in zip(dBn, dBns, dWn, dWns):
                        dB += dBs
                        dW += dWs

                # run `cross_validation` if such function is provided
                if isinstance(cross_validation, Callable):
                    total_error = 0.
                    # run forward passes on batches twice as big as `batch_size` and calculate errors
                    for i, x, y in self._get_batch(x_train, y_train, min(batch_size*2, len(x_train))):
                        if i in excluded:
                            continue
                        x = x.reshape((x.shape[0],))
                        yp = self.predict(x)
                        y = y.reshape((*y.shape, 1))
                        total_error += cross_validation(yp, y)
                    # update progress bar's legend
                    pbar.set_postfix(loss=total_error)

                for layer, dB, dW in zip(self.layers, dBn, dWn):
                    layer.update_weights(batch_size, learning_rate, dW, dB)

    def predict(self, inputs: ndarray | list[ndarray]) -> ndarray:
        "single or multiple forward pass"
        if not isinstance(inputs, list) and len(inputs.shape) == 1:
            result = [0, inputs.reshape((*inputs.shape, 1))]
            self.check_net()
            self.check_inputs([inputs])

            for layer in self.layers:
                result = layer.feed_forward(result[1])
            return result[1]
        else:
            outputs = []
            for input in inputs:
                outputs.append(list(self.predict(input)))
            return array(outputs)

    def __init__(self, input_shape: int) -> None:
        self.layers = []
        self.input_shape = input_shape

    def add_layer(self, layer: LayerModel | Layer) -> None:
        "add new layer to neural network either as model or with predefined weights and biases"
        prev_shape = self.input_shape if len(
            self.layers) == 0 else self.layers[-1].neurons
        if isinstance(layer, LayerModel):
            self.layers.append(layer.build(prev_shape))
        elif isinstance(layer, Layer):
            # check subsequent layers
            if layer.get_wshape()[0] != prev_shape:
                raise ValueError("Invalid layer provided")
            self.layers.append(layer)
        else:
            raise ValueError("Unsupported layer type")


def sigmoid(x):
    "sigmoid activation function"
    return 1./(1.+np.e**(-x))


def tanh(x):
    "tanh activation function"
    return (np.e**x-np.e**(-x))/(np.e**x+np.e**(-x))
