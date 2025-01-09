import unittest

from tona_ai.core import ActivationFunction
from tona_ai.neat import Genome


class TestGenome(unittest.TestCase):
    """
    Unit tests for the Genome class.
    """

    def setUp(self):
        """
        Set up a basic environment for testing Genome.
        """
        self.genome = Genome()

    def test_initialization(self):
        """
        Test the initialization of a Genome object.
        """
        genome = Genome()
        self.assertFalse(genome.layered)  # Genome should not be layered by default
        self.assertEqual(genome.neurons, [])
        self.assertEqual(genome.synapses, [])

    def test_create_genome_structure(self):
        """
        Test the `create` method for proper initialization of neurons and synapses.
        """
        input_size = 2
        output_size = 1
        dense = True
        layers = [[3, ActivationFunction.RELU]]
        output_activation_function = ActivationFunction.SIGMOID

        genome = self.genome.create(
            input_size=input_size,
            output_size=output_size,
            dense=dense,
            layers=layers,
            output_activation_function=output_activation_function,
        )

        # Test the structure of the genome
        self.assertEqual(len(genome.input_layer), input_size)
        self.assertEqual(len(genome.output_layer), output_size)
        self.assertEqual(
            len(genome.neurons), input_size + output_size + 3
        )  # Input + Output + Hidden
        self.assertTrue(len(genome.synapses) > 0)  # Synapses should exist

    def test_random_biases_and_weights(self):
        """
        Test that biases and weights are initialized with random values.
        """
        input_size = 2
        output_size = 2
        genome = self.genome.create(
            input_size=input_size, output_size=output_size, dense=True
        )

        # Verify that all neurons have a random bias
        for neuron in genome.neurons:
            self.assertGreaterEqual(neuron.bias, -1.0)
            self.assertLessEqual(neuron.bias, 1.0)

        # Verify that all synapses have a random weight
        for synapse in genome.synapses:
            self.assertGreaterEqual(synapse.weight, -1.0)
            self.assertLessEqual(synapse.weight, 1.0)

    def test_inheritance_from_neural_network(self):
        """
        Test that Genome inherits and uses methods from NeuralNetwork.
        """
        input_size = 3
        output_size = 2
        genome = self.genome.create(input_size=input_size, output_size=output_size)

        # Verify inheritance
        self.assertTrue(hasattr(genome, "forward"))
        self.assertTrue(callable(genome.forward))


if __name__ == "__main__":
    unittest.main()
