import unittest

from tona_ai.neat import Genome, Individual


class TestIndividual(unittest.TestCase):
    """
    Unit tests for the Individual class.
    """

    def setUp(self):
        """
        Set up a basic environment for testing Individual.
        """
        # Create a dummy genome to use for testing
        self.genome = Genome()
        self.individual = Individual(genome=self.genome)

    def test_initialization(self):
        """
        Test the initialization of an Individual object.
        """
        self.assertEqual(self.individual.genome, self.genome)
        self.assertEqual(
            self.individual.fitness, 0.0
        )  # By default, fitness should be 0

    def test_evaluate(self):
        """
        Test the evaluate method for updating the fitness of an Individual.
        """

        # Create a dummy fitness function
        def dummy_fitness_function(individual):
            return 42.0

        # Verify that the `evaluate` method updates the fitness
        fitness = self.individual.evaluate(dummy_fitness_function)
        self.assertEqual(
            fitness, 42.0
        )  # Fitness should be 42, as defined in `dummy_fitness_function`
        self.assertEqual(
            self.individual.fitness, 42.0
        )  # The individual's fitness should be updated to 42

    def test_evaluate_with_different_fitness_function(self):
        """
        Test the evaluate method with a different fitness function.
        """

        # Create another fitness function
        def alternative_fitness_function(individual):
            return 100.0

        # Verify that the `evaluate` method updates the fitness with another function
        fitness = self.individual.evaluate(alternative_fitness_function)
        self.assertEqual(fitness, 100.0)  # Fitness should be updated to 100


if __name__ == "__main__":
    unittest.main()
