import unittest

from tona_ai.core import ActivationFunction, none, relu, sigmoid, tanh


class TestActivationFunction(unittest.TestCase):
    """
    Unit tests for the ActivationFunction enum and its associated methods.
    This class validates the correctness of activation functions and their mappings.
    """

    def setUp(self):
        """
        Set up test values to be used in the activation function tests.
        """
        self.test_values = [-1.0, 0.0, 1.0]

    def test_sigmoid_activation(self):
        """
        Test that the SIGMOID activation function produces the correct outputs
        for a set of test values.
        """
        for value in self.test_values:
            expected = sigmoid(value)
            result = ActivationFunction.SIGMOID(value)
            self.assertAlmostEqual(result, expected, places=6)

    def test_relu_activation(self):
        """
        Test that the RELU activation function produces the correct outputs
        for a set of test values.
        """
        for value in self.test_values:
            expected = relu(value)
            result = ActivationFunction.RELU(value)
            self.assertAlmostEqual(result, expected, places=6)

    def test_tanh_activation(self):
        """
        Test that the TANH activation function produces the correct outputs
        for a set of test values.
        """
        for value in self.test_values:
            expected = tanh(value)
            result = ActivationFunction.TANH(value)
            self.assertAlmostEqual(result, expected, places=6)

    def test_none_activation(self):
        """
        Test that the NONE activation function produces the correct outputs
        (identity function) for a set of test values.
        """
        for value in self.test_values:
            expected = none(value)
            result = ActivationFunction.NONE(value)
            self.assertAlmostEqual(result, expected, places=6)

    def test_enum_values(self):
        """
        Verify that each enum member corresponds to the correct activation function.
        """
        self.assertEqual(ActivationFunction.SIGMOID, sigmoid)
        self.assertEqual(ActivationFunction.RELU, relu)
        self.assertEqual(ActivationFunction.TANH, tanh)
        self.assertEqual(ActivationFunction.NONE, none)


if __name__ == "__main__":
    unittest.main()
