import math


def sigmoid(x):
    """
    Computes the sigmoid activation function.

    The sigmoid function maps the input `x` to a value between 0 and 1 using the formula:
        sigmoid(x) = 1 / (1 + exp(-x))

    Args:
        x (float): Input value.

    Returns:
        float: The sigmoid of the input value.
    """
    return 1 / (1 + math.exp(-x))


def relu(x):
    """
    Computes the ReLU (Rectified Linear Unit) activation function.

    The ReLU function returns the input `x` if it is greater than 0, otherwise it returns 0:
        relu(x) = max(0, x)

    Args:
        x (float): Input value.

    Returns:
        float: The ReLU of the input value.
    """
    return max(0, x)


def tanh(x):
    """
    Computes the hyperbolic tangent (tanh) activation function.

    The tanh function maps the input `x` to a value between -1 and 1 using the formula:
        tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))

    Args:
        x (float): Input value.

    Returns:
        float: The tanh of the input value.
    """
    return math.tanh(x)


def none(x):
    """
    Returns the input value without applying any transformation.

    This can be used as a placeholder for no activation function.

    Args:
        x (float): Input value.

    Returns:
        float: The same input value `x`.
    """
    return x
