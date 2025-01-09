import timeit

from tona_ai import (
    NEAT,
    ActivationFunction,
    Environment,
    Genome,
    Individual,
    Population,
)


# Create a simple XOR environment
class XorEnvironment(Environment):
    def __init__(self):
        super().__init__()

    # Implement the run method
    # This method is called when an individual is evaluated for each generation
    def run(self, individual: Individual) -> float:
        inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
        expected_outputs = [0, 1, 1, 0]

        fitness = 0
        for index, input in enumerate(inputs):
            output = individual.genome.forward(input)
            fitness += self.fitness_calculation(
                outputs=output, expected_output=expected_outputs[index]
            )

        fitness = fitness * fitness
        individual.fitness = fitness
        return fitness

    # Implement the fitness calculation method
    def fitness_calculation(self, outputs: list[float], **kwargs: dict) -> float:
        error = (outputs[0] - kwargs["expected_output"]) ** 2
        mse = error
        fitness = 1 / (1 + mse)
        return fitness


# Create a simple genome with two inputs, one output and one hidden layer of 4 neurons
genome = Genome()
genome.create(
    input_size=2,
    output_size=1,
    layers=[
        (4, ActivationFunction.TANH),
    ],
    dense=True,
    output_activation_function=ActivationFunction.TANH,
)

# Create a population of 100 individuals based on the genome
pop = Population()
pop.create(population_size=100, initial_genome=genome)

# Create the NEAT object
neat = NEAT(
    population=pop,
    environment=XorEnvironment(),
    mutation_rate=0.1,
    mutation_range=(-0.5, 0.5),
)

start = timeit.default_timer()
# Run the NEAT algorithm for 100000 epochs
neat.run(epochs=100000)
end = timeit.default_timer()
print("Training time:", end - start, "seconds")

# Test the fitest individual
result_1 = neat.population.individuals[0].genome.forward([0.0, 0.0])
result_2 = neat.population.individuals[0].genome.forward([0.0, 1.0])
result_3 = neat.population.individuals[0].genome.forward([1.0, 0.0])
result_4 = neat.population.individuals[0].genome.forward([1.0, 1.0])

# Print the results
print("Results for the XOR gate:")
print(f"Result for 0 and 0: {result_1}, rounded: {round(result_1[0])}, expected: 0 {"OK" if round(result_1[0]) == 0 else "NOT OK"}")
print(f"Result for 0 and 1: {result_2}, rounded: {round(result_2[0])}, expected: 1 {"OK" if round(result_2[0]) == 1 else "NOT OK"}")
print(f"Result for 1 and 0: {result_3}, rounded: {round(result_3[0])}, expected: 1 {"OK" if round(result_3[0]) == 1 else "NOT OK"}")
print(f"Result for 1 and 1: {result_4}, rounded: {round(result_4[0])}, expected: 0 {"OK" if round(result_4[0]) == 0 else "NOT OK"}")
