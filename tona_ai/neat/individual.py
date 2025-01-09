from tona_ai.neat.genome import Genome


class Individual:
    """
    Represents an individual organism in the evolutionary algorithm, which includes
    a genome (a neural network) and a fitness score.

    Attributes:
        genome (Genome): The genome of the individual, representing the neural network structure.
        fitness (float): The fitness score of the individual, used to evaluate its performance.

    Methods:
        evaluate(fitness_function):
            Evaluates the fitness of the individual using a provided fitness function.
    """

    def __init__(self, genome: Genome, fitness: float = 0.0):
        """
        Initializes an Individual instance with a genome and an optional fitness score.

        Args:
            genome (Genome): The genome (neural network) of the individual.
            fitness (float, optional): The initial fitness score of the individual. Default is 0.0.
        """
        self.genome = genome
        self.fitness = fitness

    def evaluate(self, fitness_function):
        """
        Evaluates the fitness of the individual by applying a provided fitness function.

        The fitness function is expected to take an `Individual` as input and return a fitness score.

        Args:
            fitness_function (function): A function that calculates the fitness score of the individual.

        Returns:
            float: The updated fitness score of the individual.
        """
        self.fitness = fitness_function(self)
        return self.fitness
