import unittest

from tona_ai.core.activation_function import ActivationFunction
from tona_ai.core.layer import Layer
from tona_ai.core.layer_type import LayerType
from tona_ai.core.neuron_synapse import Neuron, Synapse


class TestNeuron(unittest.TestCase):
    """
    Unit tests for the Neuron class.
    """

    def setUp(self):
        """
        Set up a basic neuron and synapse configuration for testing.
        """
        # Create layers
        self.input_layer = Layer(LayerType.INPUT, 0)
        self.hidden_layer = Layer(LayerType.HIDDEN, 1)

        # Create neurons
        self.input_neuron = Neuron(
            id=1,
            layer=self.input_layer,
            bias=0.0,
            activation_function=ActivationFunction.NONE,
        )
        self.hidden_neuron = Neuron(
            id=2,
            layer=self.hidden_layer,
            bias=0.5,
            activation_function=ActivationFunction.SIGMOID,
        )

        # Create a synapse
        self.synapse = Synapse(
            in_neuron=self.input_neuron, out_neuron=self.hidden_neuron, weight=0.8
        )

        # Link the synapse to the hidden neuron
        self.hidden_neuron.inputs_synapses.append(self.synapse)

    def test_initialization(self):
        """
        Test the initialization of a Neuron object.
        """
        neuron = Neuron(
            id=3,
            layer=self.hidden_layer,
            bias=0.2,
            activation_function=ActivationFunction.TANH,
        )
        self.assertEqual(neuron.id, 3)
        self.assertEqual(neuron.layer, self.hidden_layer)
        self.assertEqual(neuron.bias, 0.2)
        self.assertEqual(neuron.activation_function, ActivationFunction.TANH)
        self.assertEqual(neuron.inputs_synapses, [])

    def test_activation_no_inputs(self):
        """
        Test the activate method when no inputs are provided.
        """
        result = self.input_neuron.activate()
        self.assertEqual(result, 0.0)  # Activation function is NONE, bias is 0.0

    def test_activation_with_inputs(self):
        """
        Test the activate method with predefined inputs.
        """
        result = self.hidden_neuron.activate(inputs=[1.0])
        expected = ActivationFunction.SIGMOID(1.0 + 0.5)  # Weighted sum + bias
        self.assertAlmostEqual(result, expected, places=6)

    def test_activation_with_memory(self):
        """
        Test the activate method with a memory dictionary.
        """
        memory = {1: 1.0}  # Memory contains the activation value of the input neuron
        result = self.hidden_neuron.activate(memory=memory)
        expected = ActivationFunction.SIGMOID(1.0 * 0.8 + 0.5)
        self.assertAlmostEqual(result, expected, places=6)
        self.assertEqual(
            memory[2], result
        )  # Ensure hidden neuron value is stored in memory

    def test_str_and_repr(self):
        """
        Test the __str__ and __repr__ methods for the Neuron class.
        """
        expected_str = f"(Neuron 2 {self.hidden_layer})"
        self.assertEqual(str(self.hidden_neuron), expected_str)
        self.assertEqual(repr(self.hidden_neuron), expected_str)


if __name__ == "__main__":
    unittest.main()
