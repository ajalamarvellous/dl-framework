"""Base functions and classes will be found here"""
import numpy as np


def neural_network(input, weight):
    input = np.array(input)
    weight = np.array(input)
    prediction = input @ weight
    return prediction
