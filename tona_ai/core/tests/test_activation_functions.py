import math
import unittest

from tona_ai.core import none, relu, sigmoid, tanh


class TestActivationFunctions(unittest.TestCase):
    """
    Unit tests for activation functions: sigmoid, relu, tanh, and none.
    """

    def setUp(self):
        """
        Set up test values for the activation function tests.
        """
        self.test_values = [-1.0, 0.0, 1.0]
        self.expected_sigmoid = [
            1 / (1 + math.exp(1)),  # sigmoid(-1.0)
            0.5,  # sigmoid(0.0)
            1 / (1 + math.exp(-1)),  # sigmoid(1.0)
        ]
        self.expected_relu = [0.0, 0.0, 1.0]  # relu(-1.0), relu(0.0), relu(1.0)
        self.expected_tanh = [
            math.tanh(-1.0),  # tanh(-1.0)
            0.0,  # tanh(0.0)
            math.tanh(1.0),  # tanh(1.0)
        ]
        self.expected_none = [-1.0, 0.0, 1.0]  # none simply returns the input

    def test_sigmoid(self):
        """
        Test the sigmoid function with a set of predefined inputs.
        """
        for i, value in enumerate(self.test_values):
            result = sigmoid(value)
            self.assertAlmostEqual(result, self.expected_sigmoid[i], places=6)

    def test_relu(self):
        """
        Test the ReLU function with a set of predefined inputs.
        """
        for i, value in enumerate(self.test_values):
            result = relu(value)
            self.assertEqual(result, self.expected_relu[i])

    def test_tanh(self):
        """
        Test the tanh function with a set of predefined inputs.
        """
        for i, value in enumerate(self.test_values):
            result = tanh(value)
            self.assertAlmostEqual(result, self.expected_tanh[i], places=6)

    def test_none(self):
        """
        Test the identity (none) function with a set of predefined inputs.
        """
        for i, value in enumerate(self.test_values):
            result = none(value)
            self.assertEqual(result, self.expected_none[i])


if __name__ == "__main__":
    unittest.main()
