from enum import Enum


class LayerType(Enum):
    """
    Enum class representing different types of layers in a neural network.

    Attributes:
        INPUT: Represents the input layer of the neural network.
        HIDDEN: Represents a hidden layer in the neural network.
        OUTPUT: Represents the output layer of the neural network.

    Methods:
        __str__():
            Returns the name of the layer type as a string.
    """

    INPUT = 0
    HIDDEN = 1
    OUTPUT = 2

    def __str__(self):
        """
        Returns the name of the layer type as a string.

        Returns:
            str: The name of the enum member (e.g., 'INPUT', 'HIDDEN', 'OUTPUT').
        """
        return self.name
