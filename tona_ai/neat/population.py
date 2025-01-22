import concurrent.futures
import copy

from tona_ai.neat.environment import Environment
from tona_ai.neat.genome import Genome
from tona_ai.neat.individual import Individual


class Population:
    """
    Represents a population of individuals (neural networks) in the evolutionary algorithm.

    Attributes:
        individuals (list[Individual]): A list of individuals (neural networks) in the population.

    Methods:
        create(population_size: int, initial_genome: Genome) -> Population:
            Creates a population of individuals by cloning the initial genome and initializing individuals.
        evaluate(environment: Environment) -> float:
            Evaluates the fitness of all individuals in the population within the given environment.
            Returns the highest fitness score in the population.
    """

    def __init__(self, individuals: list = []):
        """
        Initializes a Population instance with a list of individuals.

        Args:
            individuals (list[Individual], optional): A list of individuals (neural networks) in the population. Default is an empty list.
        """
        self.individuals = individuals

    def create(self, population_size: int = 0, initial_genome: Genome = Genome()):
        """
        Creates a population of individuals by cloning the initial genome and initializing each individual.

        Args:
            population_size (int): The number of individuals to create in the population.
            initial_genome (Genome): The initial genome to clone for each individual.

        Returns:
            Population: The population with the created individuals.
        """
        self.individuals = []
        for _ in range(population_size):
            # Clone the initial genome to avoid modifying the same object
            genome = copy.deepcopy(initial_genome)
            self.individuals.append(Individual(genome=genome))

        return self

    def evaluate(self, environment: Environment):
        """
        Evaluates the fitness of each individual in the population by running them in the provided environment.

        The population's individuals are sorted based on their fitness in descending order,
        and the highest fitness score is returned.

        Args:
            environment (Environment): The environment used to evaluate the fitness of individuals.

        Returns:
            float: The highest fitness score in the population after evaluation.
        """
        # Run each individual in the environment to evaluate their fitness
        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(environment.run, self.individuals))

        for individual, result in zip(self.individuals, results):
            individual.fitness = result or 0

        # Sort individuals by their fitness in descending order
        self.individuals.sort(key=lambda x: x.fitness, reverse=True)
        return self.individuals[0].fitness
