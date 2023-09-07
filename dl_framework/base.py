"""Base functions and classes will be found here"""
import logging

import numpy as np

logging.basicConfig(
    level=logging.DEBUG,
    filename="../.file.log",
    format="%(asctime)s %(funcName)s[%(levelname)s]: %(message)s ",
)
logger = logging.getLogger()


class NN:
    """A neural network implementation in numpy"""

    def __init__(self, input_size: int, output_size: int):
        """
        Initialising the weights for the neural networks
        input_size: size of input
        output_size: size of output
        """
        # Carefully initialise weight to be as close to zero as possible
        self._weights = np.random.normal(0, 0.1, (input_size, output_size))
        self.bias = np.random.normal(0, 0.1, 1)

    def __call__(self, input):
        self.input = input
        logger.debug(
            f"NN input shape: {self.input.shape}, weights: {self._weights.shape}"  # noqa
        )
        self.output = self.input @ self._weights + self.bias
        logger.debug(f"Output shape {self.output.shape}")
        return self.output

    def backprop(self, delta, lr):
        """Delta is the erro contribution of the nodes at the layer"""
        delta = delta.reshape(delta.shape[0], -1)
        logger.debug(
            f"Shapes input.T: {self.input.T.shape}, delta: {delta.shape}"
        )  # noqa
        # update weights here
        self._weights += self.input.T @ delta * lr
        # propagate error(delta) backwards
        return delta @ self._weights.T


class Sequential:
    """
    A sortof graph tracker to arrange the order for feedforward or Backprop
    Node(n) -> Node(n+1) -> Node(n+2)...
    stack in the queue (using python list)
    """

    def __init__(self, layers=None):
        if layers is None:
            self.layers = []
        else:
            self.layers = layers

    def __call__(self, node):
        """Add a layer (of n nodes) to the graph"""
        if isinstance(node, list):
            self.layers.extend(node)
        else:
            self.layers.append(node)

    def add(self, node):
        """Add a layer (of n nodes) to the graph"""
        if isinstance(node, list):
            self.layers.extend(node)
        else:
            self.layers.append(node)

    def predict(self, input):
        """Forward propagation through the network"""
        for layer in self.layers:
            input = layer(input)
        return input

    def backprop(self, delta, lr):
        """
        Delta is the error difference from the preceeding layer
        e.g
        ------
        delta: Array {x, n} =   y_true - y_hat or
                                chain differentiation (dy/dlx+1 * dlx+1/dx)
        """
        # invert the graph and then propagate the error(delta) backwards
        self.layers.reverse()
        for layer in self.layers:
            delta = layer.backprop(delta, lr)
        self.layers.reverse()

    def train(self, x_train, y_train, lr, epoch, batch_size, error_func):
        for _ in range(epoch):
            n = 0
            y_pred, y_true = [], []
            choices = np.random.choice(x_train.shape[0], size=batch_size)
            x_train_, y_train_ = x_train[choices, :], y_train[choices]
            for i, (X_train, Y_train) in enumerate(zip(x_train_, y_train_)):
                if len(X_train.shape) == 1:
                    X_train = np.array([X_train])
                output = self.predict(X_train)
                n += 1
                y_pred.append(output), y_true.append(Y_train)
                error = np.array(Y_train) - output
                self.backprop(error, lr)

            error = error_func(Y_train, y_pred)
            print(f"The RMSE of the model is {error}....done.")
