"""Console script for dl_framework."""
import click
import numpy as np
from activation import Relu
from base import NN, Sequential
from eval import RMSE
from optimizers import Dropout


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

    np.random.seed(2022)
    x_file = open(x_data)
    x = x_file.readlines()
    x_train = [
        [float(b.strip()) for b in a.split(",")]
        for a in x[: round(len(x) * 0.7)]  # noqa
    ]
    print("Training file processesses successfully...")
    y_file = open(y_data)
    y = y_file.read()
    y_train = [
        float(a.strip()) for a in y.split(",")[: round(len(y.split(",")) * 0.7)]  # noqa
    ]  # noqa
    print("Training labels processed successfully...")
    x_file.close(), y_file.close()

    model = Sequential()
    model([NN(5, 10), Relu(), Dropout(0.3), NN(10, 1)])

    rmse = RMSE()
    lr = 0.0001
    epoch = 100
    batch_size = 32
    print("Models parameters successfully set...")

    x_train, y_train = np.array(x_train), np.array(y_train)
    print(x_train.shape, y_train.shape)

    model.train(x_train, y_train, lr, epoch, batch_size, rmse)
    return


if __name__ == "__main__":
    main()
    # sys.exit(main())  # pragma: no cover
