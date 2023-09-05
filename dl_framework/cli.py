"""Console script for dl_framework."""
import click
import numpy as np
from activation import Relu
from base import NN
from eval import RMSE


@click.command()
@click.option("--x_data", help="training data file")
@click.option("--y_data", help="traing data label")
def main(x_data, y_data):
    """Console script for dl_framework."""
    click.echo(
        "Replace this message by putting your code into "
        "dl_framework.cli.main"  # noqa
    )
    click.echo("See click documentation at https://click.palletsprojects.com/")

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

    model = NN(5, 1)
    relu = Relu()
    # optim = GradientDescent()
    rmse = RMSE()
    alpha = 0.00001
    print("Models parameters successfully set...")

    for i, (X_train, Y_train) in enumerate(zip(x_train, y_train)):
        for _ in range(300):
            choices = np.random.choice(round(len(y_train) * 0.7), 10)
            output = model(np.array(x_train)[choices])
            print(f"Forward prop {i, _ +1} done....")  # result: {output}")
            output = relu(output)
            # print(
            # f"Correct value:{np.array(y_train)[choices]}... Output:{output}")
            error = rmse(np.array(y_train)[choices], output)
            print(f"Calculating error, Error: {error}....")
            weight_delta = (np.array(X_train) * error * alpha).reshape(-1, 1)
            model._weights -= weight_delta
            print(f"Backprop done successfully... \n {'-' * 50} \n")

    return


if __name__ == "__main__":
    main()
    # sys.exit(main())  # pragma: no cover
