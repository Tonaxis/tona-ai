import unittest

from tona_ai.neat.environment import Environment
from tona_ai.neat.genome import Genome
from tona_ai.neat.individual import Individual
from tona_ai.neat.population import Population


class TestPopulation(unittest.TestCase):
    """
    Unit tests for the Population class.
    """

    def setUp(self):
        """
        Set up a basic environment for testing the Population class.
        """
        # Create an initial genome for testing
        self.initial_genome = Genome()
        self.population = Population()

    def test_initialization(self):
        """
        Test the initialization of a Population object.
        """
        self.assertEqual(
            self.population.individuals, []
        )  # The population should be empty initially

    def test_create_population(self):
        """
        Test the create method to populate the population with individuals.
        """
        population_size = 5
        self.population.create(
            population_size=population_size, initial_genome=self.initial_genome
        )

        # Check the population size
        self.assertEqual(len(self.population.individuals), population_size)

        # Ensure each individual has a genome copied from the initial genome
        for individual in self.population.individuals:
            self.assertEqual(individual.genome.neurons, self.initial_genome.neurons)
            self.assertEqual(individual.genome.synapses, self.initial_genome.synapses)

    def test_evaluate_population(self):
        """
        Test the evaluate method which sorts individuals based on their fitness.
        """

        # Create a dummy environment class for fitness evaluation
        class DummyEnvironment(Environment):
            def __init__(self):
                self.count = 0

            def run(self, individual: Individual):
                self.count += 1
                individual.fitness = self.count

        environment = DummyEnvironment()

        # Create a population of 5 individuals
        self.population.create(population_size=5, initial_genome=self.initial_genome)

        # Evaluate the population
        highest_fitness = self.population.evaluate(environment)

        # Check that the individual with the highest fitness is ranked first
        self.assertEqual(
            highest_fitness, 5
        )  # The individual with the highest fitness will have genome ID 4

        # Verify that the individuals are sorted by fitness in descending order
        for i in range(1, len(self.population.individuals)):
            self.assertGreaterEqual(
                self.population.individuals[i - 1].fitness,
                self.population.individuals[i].fitness,
            )


if __name__ == "__main__":
    unittest.main()
