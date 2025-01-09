import random

from tona_ai.core import ActivationFunction, NeuralNetwork, Neuron, Synapse


class Genome(NeuralNetwork):
    """
    Represents a genome, which is an extension of the NeuralNetwork class.
    It inherits the functionality of a neural network, with additional features
    for initializing and evolving the weights and biases of neurons and synapses.

    Attributes:
        neurons (list[Neuron]): List of neurons in the genome (inherited from NeuralNetwork).
        synapses (list[Synapse]): List of synapses in the genome (inherited from NeuralNetwork).

    Methods:
        create(input_size: int = 0, output_size: int = 0, dense: bool = False,
               synapse_probability: float = 0.0, layers: list[int, ActivationFunction | list[ActivationFunction]] = [],
               output_activation_function: ActivationFunction | list[ActivationFunction] = ActivationFunction.NONE):
            Creates the structure of the genome, including neurons, synapses, and their weights and biases.
    """

    def __init__(self, neurons: list[Neuron] = [], synapses: list[Synapse] = []):
        """
        Initializes a Genome instance.

        Args:
            neurons (list[Neuron], optional): List of initial neurons in the genome. Default is an empty list.
            synapses (list[Synapse], optional): List of initial synapses in the genome. Default is an empty list.
        """
        super().__init__(layered=False, neurons=neurons, synapses=synapses)

    def create(
        self,
        input_size: int = 0,
        output_size: int = 0,
        dense: bool = False,
        synapse_probability: float = 0.0,
        layers: list[int, ActivationFunction | list[ActivationFunction]] = [],
        output_activation_function: (
            ActivationFunction | list[ActivationFunction]
        ) = ActivationFunction.NONE,
    ):
        """
        Creates the structure of the genome, including neurons, synapses, and their associated weights and biases.

        This method initializes neurons and synapses based on the provided parameters and assigns random biases
        to neurons and random weights to synapses. It inherits from the create method of the NeuralNetwork class.

        Args:
            input_size (int, optional): Number of input neurons. Default is 0.
            output_size (int, optional): Number of output neurons. Default is 0.
            dense (bool, optional): If True, fully connects neurons with synapses. Default is False.
            synapse_probability (float, optional): Probability of creating a synapse if not fully connected. Default is 0.0.
            layers (list[int, ActivationFunction | list[ActivationFunction]], optional): Configuration for hidden layers.
                Each layer is represented by a tuple with the number of neurons and activation function(s).
            output_activation_function (ActivationFunction | list[ActivationFunction], optional): Activation function(s)
                for the output layer. Default is `ActivationFunction.NONE`.

        Returns:
            Genome: The created genome instance with initialized neurons, synapses, and random biases and weights.
        """
        # Call the create method from the NeuralNetwork class
        super().create(
            input_size,
            output_size,
            dense,
            synapse_probability,
            layers,
            output_activation_function,
        )

        # Assign random biases to neurons
        for neuron in self.neurons:
            neuron.bias = random.uniform(-1.0, 1.0)

        # Assign random weights to synapses
        for synapse in self.synapses:
            synapse.weight = random.uniform(-1.0, 1.0)

        return self
