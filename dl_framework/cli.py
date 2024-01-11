"""Console script for dl_framework."""
import click
import cv2
import numpy as np
from activation import Relu
from base import ConvLayer, Linear, Sequential
from eval import RMSE
from optimizers import Dropout, EarlyStoppage


def prepare_data(x_data, y_data, train_size):
    x_file = open(x_data)
    x = x_file.readlines()
    x_train = [[float(b.strip()) for b in a.split(",")] for a in x]
    print("Training file processesses successfully...")

    y_file = open(y_data)
    y = y_file.read()
    y_train = [float(a.strip()) for a in y.split(",")]  # noqa
    print("Training labels processed successfully...")

    x_file.close(), y_file.close()

    idx = np.random.permutation(np.arange(len(x) - 1))
    train_idx = idx[: round(len(idx) * train_size)]
    test_idx = idx[round(len(idx) * train_size) :]  # noqa
    print(train_idx, test_idx)
    return (
        np.array(x_train)[train_idx],
        np.array(y_train)[train_idx],
        np.array(x_train)[test_idx],
        np.array(y_train)[test_idx],
    )


@click.command()
@click.option("--x_data", help="training data file")
@click.option("--y_data", help="traing data label")
def main(x_data, y_data):
    """Console script for dl_framework."""
    click.echo(
        """
        Hi there, welcome to dl-framework, a skeleton deep learning framework
        built using only numpy. This framework is meant to serve as a simple
        educational material to understand how neural networks work the way
        they do"""
        "dl_framework.cli.main"  # noqa
    )
    click.echo("See click documentation at https://click.palletsprojects.com/")

    # x_train, y_train, x_test, y_test = prepare_data(x_data, y_data, 0.75)
    x_train = x_test = np.array([cv2.imread(x_data)])
    y_train = y_test = np.array([[1]])
    # print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    np.random.seed(1)
    # , Dropout(0.3)
    # model = Sequential([Linear(5, 10), Relu(), Dropout(0.3), Linear(10, 1)])
    model = Sequential(
        [
            ConvLayer(filter_size=(3, 3), kernel_size=16, input_dim=3),
            Linear(16, 10),
            Relu(),
            Dropout(0.3),
            Linear(10, 1),
        ]
    )
    # Models can be initiated using  any of the following too
    # This
    # model = Sequential()
    # model([NN(5, 10), Relu(), Dropout(0.3), NN(10, 1)])
    # Or This
    # model = Sequential()
    # model.add([NN(5, 10), Relu(), Dropout(0.3), NN(10, 1)])

    rmse = RMSE()
    lr = 0.0001
    epoch = 300
    batch_size = 1
    early_stoppage = EarlyStoppage(patience=30)
    print("Models parameters successfully set...")

    print(x_train.shape, y_train.shape)

    model.train(
        x_train,
        y_train,
        x_test,
        y_test,
        lr,
        epoch,
        batch_size,
        rmse,
        early_stoppage,  # noqa
    )  # noqa
    return


if __name__ == "__main__":
    main()
    # sys.exit(main())  # pragma: no cover
