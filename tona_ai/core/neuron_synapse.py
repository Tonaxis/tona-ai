from __future__ import annotations

from tona_ai.core.activation_function import ActivationFunction
from tona_ai.core.layer import Layer


class Neuron:
    """
    Represents a neuron in a neural network.

    Attributes:
        id (int): Unique identifier for the neuron.
        layer (Layer): The layer to which the neuron belongs.
        bias (float): Bias value for the neuron.
        inputs_synapses (list[Synapse]): Synapses connecting to this neuron as inputs.
        activation_function (ActivationFunction): Activation function for the neuron.

    Methods:
        activate(inputs: list[float] = None, memory: dict = None) -> float:
            Activates the neuron by computing its output value based on its inputs, weights, bias, and activation function.
    """

    def __init__(
        self,
        id: int = 0,
        inputs_synapses: list[Synapse] = None,
        layer: Layer = None,
        bias: float = 0.0,
        activation_function: ActivationFunction = ActivationFunction.NONE,
    ):
        """
        Initializes a Neuron instance.

        Args:
            id (int, optional): Unique identifier for the neuron. Default is 0.
            inputs_synapses (list[Synapse], optional): Synapses providing inputs to the neuron. Default is None.
            layer (Layer, optional): Layer to which the neuron belongs. Default is None.
            bias (float, optional): Bias value for the neuron. Default is 0.0.
            activation_function (ActivationFunction, optional): Activation function for the neuron. Default is `ActivationFunction.NONE`.
        """
        self.id: int = id
        self.layer: Layer = layer
        self.bias: float = bias
        self.inputs_synapses: list[Synapse] = inputs_synapses if inputs_synapses else []
        self.activation_function: ActivationFunction = activation_function

    def activate(self, inputs: list[float] = None, memory: dict = None) -> float:
        """
        Activates the neuron by calculating its output value.

        If inputs are provided, they are used directly; otherwise, values are fetched
        recursively from the neuron's input synapses. The activation is computed as:
            output = activation_function(sum(inputs) + bias)

        Args:
            inputs (list[float], optional): Input values for the neuron. Default is None.
            memory (dict, optional): A dictionary to cache neuron activation values. Default is None.

        Returns:
            float: The output value of the neuron.
        """
        if inputs is None:
            inputs = []
            for synapse in self.inputs_synapses:
                if memory is not None and synapse.in_neuron.id in memory:
                    inputs.append(memory[synapse.in_neuron.id] * synapse.weight)
                else:
                    inputs.append(
                        synapse.in_neuron.activate(memory=memory) * synapse.weight
                    )

        value = self.activation_function(sum(inputs) + self.bias)
        if memory is not None:
            memory[self.id] = value

        return value

    def __str__(self):
        """
        Returns a string representation of the neuron.

        Returns:
            str: A string in the format "(Neuron <id> <layer>)".
        """
        return f"(Neuron {self.id} {self.layer})"

    def __repr__(self):
        """
        Returns a detailed string representation of the neuron.

        Returns:
            str: A string in the format "(Neuron <id> <layer>)".
        """
        return f"(Neuron {self.id} {self.layer})"


class Synapse:
    """
    Represents a synapse connecting two neurons in a neural network.

    Attributes:
        in_neuron (Neuron): The input neuron for the synapse.
        out_neuron (Neuron): The output neuron for the synapse.
        weight (float): The weight of the synapse.

    Methods:
        __str__():
            Returns a string representation of the synapse.
        __repr__():
            Returns a detailed string representation of the synapse.
    """

    def __init__(
        self,
        in_neuron: Neuron = None,
        out_neuron: Neuron = None,
        weight: float = 0.0,
    ):
        """
        Initializes a Synapse instance.

        Args:
            in_neuron (Neuron, optional): The input neuron. Default is None.
            out_neuron (Neuron, optional): The output neuron. Default is None.
            weight (float, optional): The weight of the synapse. Default is 0.0.
        """
        self.in_neuron: Neuron = in_neuron
        self.out_neuron: Neuron = out_neuron
        self.weight: float = weight

    def __str__(self):
        """
        Returns a string representation of the synapse.

        Returns:
            str: A string in the format "Synapse <in_neuron> -> <out_neuron>".
        """
        return f"Synapse {self.in_neuron} -> {self.out_neuron}"

    def __repr__(self):
        """
        Returns a detailed string representation of the synapse.

        Returns:
            str: A string in the format "Synapse <in_neuron> -> <out_neuron>".
        """
        return f"Synapse {self.in_neuron} -> {self.out_neuron}"
