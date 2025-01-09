import unittest

from tona_ai.core.activation_function import ActivationFunction
from tona_ai.core.layer import Layer
from tona_ai.core.layer_type import LayerType
from tona_ai.core.neuron_synapse import Neuron, Synapse


class TestSynapse(unittest.TestCase):
    """
    Unit tests for the Synapse class.
    """

    def setUp(self):
        """
        Set up test data for Synapse.
        """
        self.input_layer = Layer(LayerType.INPUT, 0)
        self.output_layer = Layer(LayerType.OUTPUT, 1)

        self.input_neuron = Neuron(
            id=1,
            layer=self.input_layer,
            bias=0.0,
            activation_function=ActivationFunction.NONE,
        )
        self.output_neuron = Neuron(
            id=2,
            layer=self.output_layer,
            bias=0.5,
            activation_function=ActivationFunction.RELU,
        )

    def test_initialization(self):
        """
        Test the initialization of a Synapse object.
        """
        synapse = Synapse(
            in_neuron=self.input_neuron, out_neuron=self.output_neuron, weight=0.8
        )
        self.assertEqual(synapse.in_neuron, self.input_neuron)
        self.assertEqual(synapse.out_neuron, self.output_neuron)
        self.assertEqual(synapse.weight, 0.8)

    def test_default_initialization(self):
        """
        Test the initialization with default parameters.
        """
        synapse = Synapse()
        self.assertIsNone(synapse.in_neuron)
        self.assertIsNone(synapse.out_neuron)
        self.assertEqual(synapse.weight, 0.0)

    def test_str_and_repr(self):
        """
        Test the __str__ and __repr__ methods for the Synapse class.
        """
        synapse = Synapse(in_neuron=self.input_neuron, out_neuron=self.output_neuron)
        expected_str = f"Synapse {self.input_neuron} -> {self.output_neuron}"
        self.assertEqual(str(synapse), expected_str)
        self.assertEqual(repr(synapse), expected_str)


if __name__ == "__main__":
    unittest.main()
