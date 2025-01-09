from abc import abstractmethod


class Environment:
    """
    An abstract base class representing the environment in which individuals (neural networks) are evaluated.

    This class defines the necessary methods that any concrete environment must implement in order to
    evaluate the performance (fitness) of individuals within that environment.

    Methods:
        run(individual, **kwargs) -> float:
            Runs the individual (neural network) within the environment and returns the resulting fitness score.
        fitness_calculation(outputs: list[float], **kwargs) -> float:
            Calculates the fitness score based on the outputs produced by the individual in the environment.
    """

    def __init__(self):
        """
        Initializes an Environment instance.
        This is the constructor for the Environment class.
        Concrete subclasses should implement specific initialization if needed.
        """
        pass

    @abstractmethod
    def run(self, individual, **kwargs: dict) -> float:
        """
        Runs the given individual (neural network) within the environment and evaluates its performance.

        Args:
            individual (Individual): The individual (neural network) to be evaluated in the environment.
            **kwargs (dict): Additional parameters passed to the environment's evaluation process.

        Returns:
            float: The fitness score of the individual after running in the environment.

        The concrete implementation of this method should simulate how the individual interacts with the environment
        and return a fitness score that represents the individual's performance.
        """
        pass

    @abstractmethod
    def fitness_calculation(self, outputs: list[float], **kwargs: dict) -> float:
        """
        Calculates the fitness score based on the outputs produced by the individual.

        Args:
            outputs (list[float]): A list of outputs produced by the individual (neural network).
            **kwargs (dict): Additional parameters passed to the fitness calculation process.

        Returns:
            float: The fitness score calculated based on the outputs.

        The concrete implementation of this method should define how to compute the fitness score
        based on the outputs produced by the individual. The fitness score can be calculated through
        any performance metric suitable for the problem domain.
        """
        pass
