from enum import Enum

from tona_ai.core.activation_functions import none, relu, sigmoid, tanh


class ActivationFunction(Enum):
    """
    Enum class representing different activation functions commonly used in neural networks.

    Attributes:
        SIGMOID: Represents the sigmoid activation function.
        RELU: Represents the ReLU (Rectified Linear Unit) activation function.
        TANH: Represents the tanh (hyperbolic tangent) activation function.
        NONE: Represents no activation function (identity function).

    Methods:
        activation(x: float) -> float:
            Applies the respective activation function to the input value `x` and returns the result.
    """

    SIGMOID = sigmoid
    RELU = relu
    TANH = tanh
    NONE = none

    def activation(self, x: float) -> float:
        """
        Applies the selected activation function to the given input value.

        Args:
            x (float): The input value to the activation function.

        Returns:
            float: The result of applying the activation function to `x`.
        """
        return self.value(x)
