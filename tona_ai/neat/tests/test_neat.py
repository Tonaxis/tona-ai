import unittest

from tona_ai.neat import NEAT, Environment, Genome, Individual, Population


class TestNEAT(unittest.TestCase):
    """
    Unit tests for the NEAT class using simplified dependencies.
    """

    def setUp(self):
        """
        Set up the environment, population, and NEAT object for testing.
        """
        # Create a simple population
        self.population = Population()

        # Create a simple environment
        self.environment = Environment()

        # Create a simple genome and individual
        self.genome = Genome()
        self.individual = Individual(self.genome)
        self.population.create(population_size=10, initial_genome=self.genome)

        # Initialize the NEAT object with the simple population and environment
        self.neat = NEAT(
            population=self.population,
            environment=self.environment,
            mutation_rate=0.2,
            mutation_range=(-0.1, 0.1),
        )

    def test_run(self):
        """
        Test the run method of NEAT.
        """
        # Simulate fitness evaluation
        self.population.evaluate(self.environment)

        # Run NEAT for a number of epochs
        epochs = 5
        self.neat.run(epochs)

        # Check that the fitness history is recorded
        self.assertEqual(len(self.neat.fitness_history), epochs)

        # Ensure that the best fitness is recorded
        best_fitness = max(ind.fitness for ind in self.population.individuals)
        self.assertEqual(self.neat.fitness_history[-1], best_fitness)

    def test_evolve(self):
        """
        Test the evolve method.
        """
        # Simulate fitness values to control selection
        self.population.individuals[0].fitness = 0.9
        self.population.individuals[1].fitness = 0.5
        self.population.individuals[2].fitness = 0.7

        # Call evolve method
        self.neat.evolve()

        # Ensure that evolution has replaced the bottom individuals with new ones
        self.assertGreater(
            self.population.individuals[0].fitness, 0.0
        )  # Top individual
        self.assertGreater(
            self.population.individuals[1].fitness, 0.0
        )  # Replaced individual

    def test_mutate(self):
        """
        Test the mutate method.
        """
        # Create an individual with a genome
        individual = Individual(self.genome)

        # Set mutation rate to 1.0 to force mutation
        self.neat.mutation_rate = 1.0

        # Simulate mutation
        self.neat.mutate(individual)

        # Ensure that synapse weights and neuron biases are mutated
        for synapse in individual.genome.synapses:
            self.assertGreater(synapse.weight, 0.0)

        for neuron in individual.genome.neurons:
            self.assertGreater(neuron.bias, 0.0)

    def test_mutate_population(self):
        """
        Test the mutate_population method.
        """
        # Set mutation rate to 1.0 to force mutation
        self.neat.mutation_rate = 1.0

        # Mutate the population
        self.neat.mutate_population()

        # Ensure that all individuals' genomes have been mutated
        for individual in self.population.individuals:
            for synapse in individual.genome.synapses:
                self.assertGreater(synapse.weight, 0.0)

            for neuron in individual.genome.neurons:
                self.assertGreater(neuron.bias, 0.0)


if __name__ == "__main__":
    unittest.main()
