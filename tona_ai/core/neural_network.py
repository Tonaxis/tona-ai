import pickle
import random

from tona_ai.core.activation_function import ActivationFunction
from tona_ai.core.layer import Layer
from tona_ai.core.layer_type import LayerType
from tona_ai.core.neuron_synapse import Neuron, Synapse


class NeuralNetwork:
    """
    Represents a neural network, including neurons, synapses, and optional layer organization.

    Attributes:
        layered (bool): Whether the neural network is organized into layers.
        neurons (list[Neuron]): List of neurons in the network.
        synapses (list[Synapse]): List of synapses in the network.
        input_layer (list[Neuron]): Neurons in the input layer.
        output_layer (list[Neuron]): Neurons in the output layer.
        input_size (int): Number of input neurons.
        output_size (int): Number of output neurons.
        last_neuron_id_added (int): Tracks the ID of the last added neuron.
    """

    def __init__(
        self, layered: bool, neurons: list[Neuron] = [], synapses: list[Synapse] = []
    ):
        """
        Initializes the neural network.

        Args:
            layered (bool): Indicates if the network is organized into layers.
            neurons (list[Neuron], optional): List of initial neurons. Default is an empty list.
            synapses (list[Synapse], optional): List of initial synapses. Default is an empty list.
        """
        self.layered: bool = layered
        self.neurons: list[Neuron] = neurons
        self.synapses: list[Synapse] = synapses
        self.input_layer: list[Neuron] = []
        self.output_layer: list[Neuron] = []
        self.input_size: int = 0
        self.output_size: int = 0
        self.last_neuron_id_added: int = (
            max(neuron.id for neuron in self.neurons) if self.neurons else 0
        )

        for neuron in self.neurons:
            if neuron.layer.layer_type == LayerType.INPUT:
                self.input_layer.append(neuron)
                self.input_size += 1
            elif neuron.layer.layer_type == LayerType.OUTPUT:
                self.output_layer.append(neuron)
                self.output_size += 1

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
        Creates the structure of the neural network.

        Args:
            input_size (int, optional): Number of input neurons. Default is 0.
            output_size (int, optional): Number of output neurons. Default is 0.
            dense (bool, optional): If True, fully connects neurons with synapses. Default is False.
            synapse_probability (float, optional): Probability of creating a synapse if not dense. Default is 0.0.
            layers (list[int, ActivationFunction | list[ActivationFunction]], optional): Hidden layer structure.
                Each layer is represented by a tuple with the number of neurons and activation function(s).
            output_activation_function (ActivationFunction | list[ActivationFunction], optional): Activation function(s)
                for the output layer neurons. Default is `ActivationFunction.NONE`.
        """
        self.input_size = input_size
        self.output_size = output_size
        self.input_layer = []
        self.output_layer = []
        self.neurons = []
        self.synapses = []
        self.last_neuron_id_added = 0

        # Adding input neurons
        for _ in range(input_size):
            self.input_layer.append(self._add_neuron(layer=LayerType.INPUT))

        # Adding output neurons
        for i in range(output_size):
            activation_function = (
                output_activation_function[i]
                if isinstance(output_activation_function, list)
                else output_activation_function
            )
            self.output_layer.append(
                self._add_neuron(
                    layer=LayerType.OUTPUT, activation_function=activation_function
                )
            )

        # Adding hidden neurons
        added_hidden_layers = []
        for layer in layers:
            added_hidden_layer = []
            for i in range(layer[0]):
                activation_function = (
                    layer[1][i] if isinstance(layer[1], list) else layer[1]
                )
                added_hidden_layer.append(
                    self._add_neuron(
                        layer=LayerType.HIDDEN, activation_function=activation_function
                    )
                )
            added_hidden_layers.append(added_hidden_layer)

        # Creating synapses
        possible_synapses = []
        left_layer = self.input_layer
        hidden_index = -1

        for _ in range(len(added_hidden_layers) + 1):
            right_layer = (
                added_hidden_layers[hidden_index + 1]
                if len(added_hidden_layers) > hidden_index + 1
                else self.output_layer
            )
            hidden_index += 1

            for left_neuron in left_layer:
                for right_neuron in right_layer:
                    possible_synapses.append(
                        Synapse(in_neuron=left_neuron, out_neuron=right_neuron)
                    )

            left_layer = right_layer

        # Adding synapses to the network
        for synapse in possible_synapses:
            if dense or random.random() < synapse_probability:
                synapse.weight = random.uniform(-1.0, 1.0)
                self.synapses.append(synapse)
                synapse.out_neuron.inputs_synapses.append(synapse)

    def forward(self, inputs: list[float]) -> list[float]:
        """
        Performs a forward pass through the network.

        Args:
            inputs (list[float]): Input values for the network.

        Returns:
            list[float]: Output values from the network.
        """
        memory: dict = {}
        outputs: list[float] = []

        for index, input_value in enumerate(inputs):
            memory[self.input_layer[index].id] = input_value

        for output in self.output_layer:
            outputs.append(output.activate(memory=memory))

        return outputs

    def _add_neuron(
        self,
        layer: LayerType,
        layer_index: int = 0,
        bias: float = 0.0,
        activation_function: ActivationFunction = ActivationFunction.NONE,
    ) -> Neuron:
        """
        Adds a neuron to the network.

        Args:
            layer (LayerType): The type of the layer to which the neuron belongs.
            layer_index (int, optional): The index of the layer. Default is 0.
            bias (float, optional): The bias value of the neuron. Default is 0.0.
            activation_function (ActivationFunction, optional): Activation function for the neuron. Default is `ActivationFunction.NONE`.

        Returns:
            Neuron: The created neuron.
        """
        neuron = Neuron(
            id=self.last_neuron_id_added,
            layer=Layer(layer, layer_index),
            bias=bias,
            activation_function=activation_function,
        )
        self.neurons.append(neuron)
        self.last_neuron_id_added += 1
        return neuron

    def save(self, filename: str):
        """Saves the neural network to a file.

        Args:
            filename (str): The name of the file to save the network to.
        """
        with open(filename, "wb") as file:
            pickle.dump(self, file)

    def load(filename: str) -> "NeuralNetwork":
        """Loads a neural network from a file.

        Args:
            filename (str): The name of the file to load the network from.

        Returns:
            NeuralNetwork: The loaded neural network.
        """
        with open(filename, "rb") as file:
            return pickle.load(file)
