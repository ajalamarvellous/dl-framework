"""Base functions and classes will be found here"""
import logging

import numpy as np

logging.basicConfig(
    level=logging.DEBUG,
    filename="../.file.log",
    format="%(asctime)s %(funcName)s[%(levelname)s]: %(message)s ",
)
logger = logging.getLogger()


class Linear:
    """A neural network implementation in numpy"""

    def __init__(self, input_size: int, output_size: int):
        """
        Initialising the weights for the neural networks
        input_size: size of input
        output_size: size of output
        """
        # Carefully initialise weight to be as close to zero as possible
        self._weights = np.random.normal(0, 0.1, (input_size, output_size))
        self._bias = np.random.normal(0, 0.1, 1)

    def __call__(self, input):
        self._input = input
        logger.debug(
            f"NN input shape: {self._input.shape}, weights: {self._weights.shape}"  # noqa
        )
        # dot product of the inputs and weight
        self._output = self.input @ self._weights + self._bias
        logger.debug(f"Output shape {self._output.shape}")
        return self._output

    def backprop(self, delta, lr):
        """Delta is the error contribution of the nodes at the layer"""
        # reshape the delta (back propagated error) into shape of output
        delta = delta.reshape(delta.shape[0], -1)
        logger.debug(
            f"Shapes input.T: {self._input.T.shape}, delta: {delta.shape}"
        )  # noqa
        # update weights here
        self._weights += self._input.T @ delta * lr
        # propagate error(delta) backwards by multiplying the delta by
        # gradient of the layer d/dx(w * x + b) = w, thus
        # delta * weight = product of all propagated errors till that layer
        return delta @ self._weights.T


class ConvLayer:
    def __init__(
        self, filter_size, kernel_size, input_dim, stride=(1, 1), padding=(0, 0)  # noqa
    ):  # noqa
        self._input_dim = input_dim  # no of dimensions of the image (1 or 3)
        self._filter_size = filter_size  # size of fikler for sliding e.g(3, 3)
        self._kernel_size = kernel_size  # size of kernel to transform images
        # kernel weights of the dimension of the filter_size
        # input_size = flattened filter size
        # output = desired no of kernel transformation
        self._kernel_weights = Linear(
            self.filter_size[0] * self.filter_size[1], self.kernel_size
        )
        # size of the movements to make (x, y)
        self._stride = stride
        # padding to add to the image to keep the image in the same dimensions
        # TODO: Add oadding
        self._padding = padding

    def _get_image_chunks(self, image):
        dims = image.shape  # get images dimensions
        image_chunk = []
        # consider this as sliding over the image by the dimension of th filter
        # and creating a chunk or fragment of all the unique vies
        for x_i in range(dims[1] - self._filter_size[0], self._stride[0]):
            for y_i in range(dims[2] - self._filter_size[1], self._stride[1]):
                # images coming in dimension
                # (no_images, x[height], y[width], z(image dimensions))
                chunk = image[
                    :,
                    x_i : x_i + self._filter_size[0],  # noqa
                    y_i : y_i + self._filter_size[1],  # noqa
                    :,  # noqa
                ].reshape(
                    # reshape image to (no_images, dims, x, y)
                    -1,
                    dims[-1],
                    self._filter_size[0],
                    self._filter_size[1],
                )  # noqa
                image_chunk.append(chunk)
        # concatenate into a long row of filter chunks
        expanded = np.concatenate(image_chunk, axis=0)
        # flatten to have each chunk as a row vector
        flattened_input = expanded.reshape(
            -1, self._filter_size[0] * self._filter_size[1]
        )
        return flattened_input

    def __call__(self, images):
        # forward prop
        return self._kernel(self._get_image_chunks(images))

    def backprop(self, delta, lr):
        """Delta is the erro contribution of the nodes at the layer"""
        return self._kernel.backprop(delta, lr)


class Sequential:
    """
    A sortof graph tracker to arrange the order for feedforward or Backprop
    Node(n) -> Node(n+1) -> Node(n+2)...
    stack in the queue (using python list)
    """

    def __init__(self, layers=None):
        if layers is None:
            self._layers = []
        else:
            self._layers = layers

    def __call__(self, node):
        """Add a layer (of n nodes) to the graph"""
        if isinstance(node, list):
            self._layers.extend(node)
        else:
            self._layers.append(node)

    def add(self, node):
        """Add a layer (of n nodes) to the graph"""
        if isinstance(node, list):
            self._layers.extend(node)
        else:
            self._layers.append(node)

    def predict(self, input):
        """Forward propagation through the network"""
        for layer in self._layers:
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
        self._layers.reverse()
        for layer in self.layers:
            delta = layer.backprop(delta, lr)
        self._layers.reverse()

    def train(
        self,
        x_train,
        y_train,
        x_test,
        y_test,
        lr,
        epoch,
        batch_size,
        error_func,
        early_stoppage=None,
    ):
        for _ in range(epoch):
            y_pred, y_true = [], []
            if batch_size == 1:
                x_train_, y_train_ = x_train, y_train
                x_test_, y_test_ = x_test, y_test
            else:
                choices = np.random.choice(x_train.shape[0], size=batch_size)
                x_train_, y_train_ = x_train[choices, :], y_train[choices]
                x_test_, y_test_ = x_test[choices, :], y_test[choices, :]

            for X_train, Y_train in zip(x_train_, y_train_):
                if len(X_train.shape) == 1:
                    X_train = np.array([X_train])
                output = self.predict(X_train)
                y_pred.append(output), y_true.append(Y_train)
                error = np.array(Y_train) - output
                self.backprop(error, lr)

            y_pred = np.array(y_pred).flatten()
            y_true = np.array(y_true).flatten()
            train_error = error_func(y_true, y_pred)

            y_pred_test = self.predict(x_test_).flatten()
            test_error = error_func(np.array(y_test_), y_pred_test)

            print(f"Epoch {_}/ {epoch} done...")
            if early_stoppage is not None:
                early_stoppage(self.layers, test_error)
                if early_stoppage.early_stoppage is True:  # noqa
                    break
            print(
                f"The train error: {train_error}, test error: {test_error}...."
            )  # noqa

        self._layers = early_stoppage.model
