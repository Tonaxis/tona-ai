import unittest

from tona_ai.core import (
    ActivationFunction,
    Layer,
    LayerType,
    NeuralNetwork,
    Neuron,
    Synapse,
)


class TestNeuralNetwork(unittest.TestCase):
    """
    Unit tests for the NeuralNetwork class.
    """

    def setUp(self):
        """
        Set up test data for the NeuralNetwork class.
        """
        self.neuron1 = Neuron(
            id=1,
            layer=Layer(LayerType.INPUT, 0),
            bias=0.1,
            activation_function=ActivationFunction.SIGMOID,
        )
        self.neuron2 = Neuron(
            id=2,
            layer=Layer(LayerType.OUTPUT, 0),
            bias=0.2,
            activation_function=ActivationFunction.RELU,
        )
        self.synapse = Synapse(
            in_neuron=self.neuron1, out_neuron=self.neuron2, weight=0.5
        )

        self.nn = NeuralNetwork(
            layered=False, neurons=[self.neuron1, self.neuron2], synapses=[self.synapse]
        )

    def test_initialization(self):
        """
        Test the initialization of a NeuralNetwork object.
        """
        self.assertEqual(self.nn.layered, False)
        self.assertEqual(len(self.nn.neurons), 2)
        self.assertEqual(len(self.nn.synapses), 1)
        self.assertEqual(len(self.nn.input_layer), 1)
        self.assertEqual(len(self.nn.output_layer), 1)

    def test_add_neuron(self):
        """
        Test the _add_neuron method to ensure neurons are correctly added.
        """
        neuron = self.nn._add_neuron(
            layer=LayerType.HIDDEN,
            layer_index=1,
            bias=0.3,
            activation_function=ActivationFunction.TANH,
        )
        self.assertEqual(neuron.id, 2)
        self.assertEqual(neuron.layer.layer_type, LayerType.HIDDEN)
        self.assertEqual(neuron.bias, 0.3)
        self.assertEqual(neuron.activation_function, ActivationFunction.TANH)
        self.assertEqual(len(self.nn.neurons), 3)

    def test_create_dense_network(self):
        """
        Test the creation of a dense neural network.
        """
        self.nn.create(input_size=2, output_size=1, dense=True)
        self.assertEqual(len(self.nn.input_layer), 2)
        self.assertEqual(len(self.nn.output_layer), 1)
        self.assertTrue(all(synapse.weight != 0 for synapse in self.nn.synapses))
        self.assertGreater(len(self.nn.synapses), 0)

    def test_create_sparse_network(self):
        """
        Test the creation of a sparse neural network with synapse probability.
        """
        self.nn.create(
            input_size=2, output_size=1, dense=False, synapse_probability=0.5
        )
        self.assertEqual(len(self.nn.input_layer), 2)
        self.assertEqual(len(self.nn.output_layer), 1)
        self.assertLessEqual(
            len(self.nn.synapses), len(self.nn.input_layer) * len(self.nn.output_layer)
        )

    def test_forward_pass(self):
        """
        Test the forward method with predefined inputs.
        """
        self.nn.create(
            input_size=1,
            output_size=1,
            output_activation_function=ActivationFunction.NONE,
            dense=True,
        )
        self.nn.neurons[0].bias = 0.0
        self.nn.neurons[1].bias = 0.2
        self.nn.synapses[0].weight = 0.5
        outputs = self.nn.forward([1.0])
        self.assertEqual(outputs, [0.7])

    def test_synapse_creation(self):
        """
        Test that synapses are correctly created between neurons.
        """
        self.nn.create(input_size=1, output_size=1, dense=True)
        self.assertEqual(len(self.nn.synapses), 1)
        self.assertEqual(self.nn.synapses[0].in_neuron, self.nn.input_layer[0])
        self.assertEqual(self.nn.synapses[0].out_neuron, self.nn.output_layer[0])

    def test_neural_network_save_and_load(self):
        """
        Test the save and load methods for the NeuralNetwork class.
        """
        self.nn.save("test_network.pkl")
        loaded_nn = NeuralNetwork.load("test_network.pkl")
        self.assertEqual(len(loaded_nn.neurons), len(self.nn.neurons))
        self.assertEqual(len(loaded_nn.synapses), len(self.nn.synapses))


if __name__ == "__main__":
    unittest.main()
