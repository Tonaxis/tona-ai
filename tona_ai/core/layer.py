from tona_ai.core.layer_type import LayerType


class Layer:
    """
    Represents a layer in a neural network.

    Attributes:
        layer_type (LayerType): The type of the layer (e.g., INPUT, HIDDEN, OUTPUT).
        layer_index (int): The index of the layer in the network (default is 0).

    Methods:
        __str__():
            Returns a string representation of the layer, including its type and index.
    """

    def __init__(self, layer_type: LayerType, layer_index: int = 0):
        """
        Initializes a Layer instance.

        Args:
            layer_type (LayerType): The type of the layer.
            layer_index (int, optional): The index of the layer in the network. Default is 0.
        """
        self.layer_type = layer_type
        self.layer_index = layer_index

    def __str__(self):
        """
        Returns a string representation of the layer.

        The representation includes the layer type and its index, formatted as:
            "<layer_type> <layer_index>"

        Returns:
            str: A string representing the layer.
        """
        return f"{self.layer_type} {self.layer_index}"
