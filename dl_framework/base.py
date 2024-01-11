"""Base functions and classes will be found here"""
import logging

import numpy as np

logging.basicConfig(
    level=logging.DEBUG,
    # filename="../.file.log",
    format="%(asctime)s %(funcName)s[%(levelname)s]: %(message)s ",
)
logger = logging.getLogger()


class Linear:
    """
    A neural network node implementation in numpy
    y = wx + b
    where:
        w = weights
        x = input data
        b = bias

    Usage
    --------
    from base import Linear

    INPUT_DIM = 10
    OUTPUT_DIM = 1
    lr = 0.01  # Learning rate

    model = Linear(INPUT_DIM, OUTPUT_DIM)

    # forward propagation
    y_pred = model(input)

    # backward propagation
    error = error_func(y_true, y_pred)

    model.backprop(error, lr)

    it can also be used in Sequential

    model = Sequential([
        Linear(INPUT_DIM, INPUT_DIM), Linear(INPUT_DIM, OUTPUT_DIM)
    ])
    ...
    """

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
        """
        Forward call

        Parameter(s)
        -------------
        input : np.array(...)
            data propagating forward

        Return(s)
        ----------
        y : np.array(...)
            y is the dot product of input data and layer weights
        """
        self._input = input
        logger.debug(
            f"NN input shape: {self._input.shape}, weights: {self._weights.shape}"  # noqa
        )
        assert (
            self._input.shape[1] == self._weights.shape[0]
        ), f"\
            The dimensions don't match for, input: {self._input.shape[1]} \
            weights: {self._weights.shape[0]}, \n\
            expects input of dims ({self._input.shape[1], self._weights.shape[0]})"  # noqa
        # dot product of the inputs and weight
        self._output = self._input @ self._weights + self._bias
        logger.debug(f"Output shape {self._output.shape}")
        return self._output

    def backprop(self, delta, lr):
        """
        Backpropagation call

        Parameter(s)
        ------------------
        delta : np.array(...) | float
            Delta is the error contribution of the nodes at the layer
        lr : float
            learning rate, determining how big the optimisation step should be

        Return(s)
        -----------
        delta: np.array(...) | float
            gradient of the error till that layer
            d/dx(w * x + b) = w
            using chain rule to get the error from the last layer
            delta = delta * weight

        Other(s)
        -----------
        layer weights are also updated
            layer inputs (transpose) * delta (propagated to that layer)
        """
        # reshape the delta (back propagated error) into shape of output
        delta = delta.reshape(delta.shape[0], -1)
        logger.debug(
            f"Shapes input.T: {self._input.T.shape}, delta: {delta.shape}"
        )  # noqa
        # update weights here
        self._weights += self._input.T @ delta * lr
        return delta @ self._weights.T


class ConvLayer:
    def __init__(
        self, filter_size, kernel_size, input_dim, stride=(1, 1), padding=(0, 0)  # noqa
    ):  # noqa
        # no of dimensions of the image (1 or 3)
        self._input_dim = input_dim
        # size of fikler for sliding e.g(3, 3)
        self._filter_size = filter_size
        # size of kernel to transform images
        self._kernel_size = kernel_size
        # kernel weights of the dimension of the filter_size
        # input_size = flattened filter size
        # output = desired no of kernel transformation
        self._kernel = Linear(
            self._filter_size[0] * self._filter_size[1], self._kernel_size
        )
        # size of the movements to make (x, y)
        self._stride = stride
        # padding to add to the image to keep the image in the same dimensions
        # TODO: Add oadding
        self._padding = padding

    def _get_image_chunks(self, image):
        dims = image.shape  # get images dimensions
        logger.debug(f"Image shape {dims}")
        if len(dims) == 3:
            image = np.array([image])
            self._get_image_chunks(image)
        image_chunk = []
        # consider this as sliding over the image by the dimension of th filter
        # and creating a chunk or fragment of all the unique vies
        for x_i in range(0, dims[1] - self._filter_size[0], self._stride[0]):
            for y_i in range(
                0, dims[2] - self._filter_size[1], self._stride[1]
            ):  # noqa
                # images coming in dimension
                # (no_images, x[height], y[width], z(image dimensions))
                chunk = image[
                    :,
                    x_i : x_i + self._filter_size[0],  # noqa
                    y_i : y_i + self._filter_size[1],  # noqa
                    :,  # noqa
                ]
                # ].reshape(
                #     # reshape image to (no_images, dims, x, y)
                #     -1,
                #     dims[-1],
                #     self._filter_size[0],
                #     self._filter_size[1],
                # )  # noqa
                image_chunk.append(chunk)
        logger.info(
            f"Image chunk shape: {len(image_chunk)}, {image_chunk[0].shape}"
        )  # noqa
        # concatenate into a long row of filter chunks
        expanded = np.array(image_chunk)
        # expanded = np.concatenate(image_chunk, axis=0)
        logger.info(f"Expanded images dims: {expanded.shape}")
        # flatten to have each chunk as a row vector
        flattened_input = expanded.reshape(
            -1, self._filter_size[0] * self._filter_size[1]
        )
        logger.info(f"Flattened image shape: {flattened_input.shape}")
        return flattened_input

    def __call__(self, images):
        # forward prop
        images_chunk = self._get_image_chunks(images)
        return self._kernel(images_chunk)

    def backprop(self, delta, lr):
        """Delta is the erro contribution of the nodes at the layer"""
        return self._kernel.backprop(delta, lr)


class Sequential:
    """
    A sortof graph tracker to arrange the order for feedforward or Backprop
    Node(n) -> Node(n+1) -> Node(n+2)...stack in the queue (using python list)

    Usage
    --------
    from base import Linear, Sequential

    INPUT_DIM = 10
    OUTPUT_DIM = 1
    lr = 0.01  # Learning rate

    model = Sequential([
        Linear(INPUT_DIM, INPUT_DIM), Linear(INPUT_DIM, OUTPUT_DIM)
    ])

    # forward propagation
    y_pred = model.predict(input)

    # backward propagation
    error = error_func(y_true, y_pred)

    model.backprop(error, lr)

    it also has a method that performs complete forward and backprop for a
    specified number of epochs
    model.train(
        x_train,
        y_train,
        x_test,
        y_test,
        lr,
        epoch,
        batch_size,
        error_func,
        early_stoppage
    )
    ...
    """

    def __init__(self, layers=None):
        """Iniatilising the sequential list"""
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
        """
        Forward propagation through the network
        calls the model.__call__() method of every layer in it list and
        propagate the output of the previous layer as input to the next layer
        """
        for layer in self._layers:
            input = layer(input)
        return input

    def backprop(self, delta, lr):
        """
        Backpropagation of the error through all the layers in the Model by
        call each layers .backprop() method

        backprop first reverse self.layers (self.layers[start,end, -1]),
        it then send chains all the layers error together internally using the
        layers .backprop() such that the delta from that layer is the product
        of the previous layers gradient multiplied by the gradient of that
        layer and then update each of those layers.

        Activation functions also recieve the lr majorly because of this method
        as this method is respondible for getting the gradient of each layer as
        well as updating them and given that the Linear layers require lr,
        thus, the activation functions are made to recieve the lr too for api
        consistency

        Parameter(s)
        ---------------
        delta : np.array(...) | float
            delta is the error contribution to be propagated through the entire
            network
        lr : float
            learning rate, determining how big the optimisation step should be

        Return(s)
        -----------
        delta: np.array(...) | float
            gradient of the error till that layer
            delta = delta * weight
        """
        # invert the graph and then propagate the error(delta) backwards
        self._layers.reverse()
        for layer in self._layers:
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
        """
        A complete forward and backward propagation of the entire network

        Parameter(s)
        --------------
        x_train : np.array(...)
            training data
        y_train : np.array(...)
            training labels
        x_test : np.array(...)
            testing data
        y_test : np.array(...)
            testing labels
        lr : float
            learning rate (how big size of learning update should be)
        epoch : int
            number of training epochs
        batch_size :
            size of batch to train on for every epoch
        error_func : Function()
            Function to calculate the error of the model
        early_stoppage : Function()
            Function to stop the model from training again after no successive
            improvement is observed in the model.
        """
        for _ in range(epoch):
            y_pred, y_true = [], []
            # if the batch_size is set to one, it means use the whole Training
            # data for every batch
            if batch_size == 1:
                x_train_, y_train_ = x_train, y_train
                x_test_, y_test_ = x_test, y_test
            else:
                # randomly select the number of batch specified to train that
                # epoch
                choices = np.random.choice(x_train.shape[0], size=batch_size)
                x_train_, y_train_ = x_train[choices, :], y_train[choices]
                x_test_, y_test_ = x_test[choices, :], y_test[choices, :]

            for X_train, Y_train in zip(x_train_, y_train_):
                # because the model can propagate many samples together or
                # just one, however the shape will differ (multiple samples
                # together will have a (m,n) shape while just one will have
                # just (n,) shape, so reshape one training sampple to (1,n)
                # consistency sake
                if len(X_train.shape) == 1:
                    X_train = np.array([X_train])
                # forward propagation
                output = self.predict(X_train)
                # get the predictions and correct value
                y_pred.append(output), y_true.append(Y_train)
                # calculate the difference between each correct and predicted
                # value, we need this for the Backpropagation, for the model
                # to account for error contribution to each of the layers
                error = np.array(Y_train) - output
                # backpropagate error
                self.backprop(error, lr)

            # get error summation via error function
            y_pred = np.array(y_pred).flatten()
            y_true = np.array(y_true).flatten()
            train_error = error_func(y_true, y_pred)

            # evaluate model on test set
            y_pred_test = self.predict(x_test_).flatten()
            test_error = error_func(np.array(y_test_), y_pred_test)

            print(f"Epoch {_}/ {epoch} done...")
            # check if model is not improving and stop the model if
            # early_stoppage is activated
            if early_stoppage is not None:
                early_stoppage(self._layers, test_error)
                if early_stoppage.early_stoppage is True:  # noqa
                    break
            print(
                f"The train error: {train_error}, test error: {test_error}...."
            )  # noqa

        # set the model weights to the best weight saved by ealy stoppage
        self._layers = early_stoppage.model
