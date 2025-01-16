<div align="center">

<a href="https://github.com/tonaxis/tona-ai">
<!-- <img media="(prefers-color-scheme: dark)" src="./docs/images/tona_ai_logo_dark.svg" alt="Logo of Tona AI" width="450px"> -->
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./docs/images/tona_ai_logo_dark.svg" alt="Logo of Tona AI" width="450px">
  <source media="(prefers-color-scheme: light)" srcset="./docs/images/tona_ai_logo_light.svg" alt="Logo of Tona AI" width="450px">
  <img alt="Logo of Tona AI" width="450px" src="./docs/images/tona_ai_logo_light.svg">
</picture>
</a>

**Make Artificial Intelligence easier**

![Static Badge](https://img.shields.io/badge/BETA-Status?label=Status&color=yellow)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![GitHub release](https://img.shields.io/github/v/release/tonaxis/tona-ai)](https://github.com/tonaxis/tona-ai/releases)
![](https://img.shields.io/badge/Python-3.12.7-3776AB?style=flat-circle&logo=python&logoColor=3776AB)
</div>

## Summary
- [Introduction](#introduction)
- [Installation](#installation)
- [Neural Network](#neural-network)
    - [Explanations](#explanations)
    - [Usage](#usage)
        - [Create a neural network](#create-a-neural-network)
        - [Execute a neural network](#execute-a-neural-network)
        - [Save a neural network](#save-a-neural-network)
        - [Load a neural network](#load-a-neural-network)
- [NEAT (WIP)](#neat)
    - [Disclaimer](#disclaimer)
    - [Explanations](#explanations-1)
    - [Usage](#usage-1)
        - [Setup an Envirnment](#setup-an-environment)
        - [Execute the NEAT algorithm](#execute-the-neat-algorithm)

## Introduction
This package was created to facilitate the manipulation of neural networks, understand their functioning, create custom models, and train them. It is developed in Python entirely from scratch, without using external packages beyond Python.
> Note: This project is currently in Beta and still under active development. Many features are planned for future releases.

## Installation
```shell
pip install tona-ai
```

## Neural Network
### Explanations
A **neural network** consists of neurons and synapses. The neurons are organized into different layers:
- The input layer ``INPUT``
    - This layer is mandatory and unique as it is the neurons in this layer that receive the input values of the neural network when it is executed.
- The hidden layers ``HIDDEN``
    - These layers are optional and there can be several of them. They allow the neural network to perform more complex calculations and be more precise.
- The output layer ``OUTPUT``
    - This layer is mandatory and unique. The neurons in this layer receive the output values of the neural network when it is executed.

Neurons have an activation function and a bias. The activation function calculates the neuron's output, and the bias is added to the sum of the inputs before passing through the activation function. This bias adjusts the output.

Available activation functions are:
- Sigmoid function
- ReLU (Rectified Linear Unit)
- Hyperbolic tangent (TANH)

Synapses serve as connections between neurons. Each synapse has an input neuron and an output neuron. Synapses also have a weight that can be adjusted and will be multiplied by the input neuron's output before being sent to the output neuron.

Tona AI allows you to create a custom neural network. You have access to each **neuron** and each **synapse**.

### Usage
#### Create a neural network
To create a neural network, you can simply use the ``create()`` method of the **NeuralNetwork** class:
```python
from tona_ai import ActivationFunction, NeuralNetwork

# Layered parameter is required, to define if the network is organized into strict layers
nn = NeuralNetwork(layered=False)
nn.create(
    input_size=2,
    output_size=1,
    layers=[
        (4, ActivationFunction.TANH),
    ],
    dense=True,
    output_activation_function=ActivationFunction.TANH,
)
```
Example code available [here](./examples/creation_neural_network.py)!

Or you can manually create the neurons and the connections between them:
```python
from tona_ai import (
    ActivationFunction,
    Layer,
    LayerType,
    NeuralNetwork,
    Neuron,
    Synapse,
)

# Define the layers
input_layer = Layer(layer_type=LayerType.INPUT)
output_layer = Layer(layer_type=LayerType.OUTPUT)
# hidden layer can have multiple layers so you need to specify the index
hidden_layer = Layer(layer_type=LayerType.HIDDEN, layer_index=0)

# Define the neurons
neurons = [
    # Input Neurons do not have a bias or activation
    Neuron(id=0, layer=input_layer),  # x1
    Neuron(id=1, layer=input_layer),  # x2
    # Hidden Neurons
    Neuron(
        id=2,
        layer=hidden_layer,
        bias=0.0,
        activation_function=ActivationFunction.RELU,
    ),  # h1
    Neuron(
        id=3,
        layer=hidden_layer,
        bias=0.0,
        activation_function=ActivationFunction.RELU,
    ),  # h2
    # Output Neurons
    Neuron(
        id=4,
        layer=output_layer,
        bias=-2.0,
        activation_function=ActivationFunction.SIGMOID,
    ),  # o
]

# Define the synapses
synapses = [
    Synapse(in_neuron=neurons[0], out_neuron=neurons[2], weight=2.0),  # x1 -> h1
    Synapse(in_neuron=neurons[0], out_neuron=neurons[3], weight=-2.0),  # x1 -> h2
    Synapse(in_neuron=neurons[1], out_neuron=neurons[2], weight=-2.0),  # x2 -> h1
    Synapse(in_neuron=neurons[1], out_neuron=neurons[3], weight=2.0),  # x2 -> h2
    Synapse(in_neuron=neurons[2], out_neuron=neurons[4], weight=2.0),  # h1 -> o
    Synapse(in_neuron=neurons[3], out_neuron=neurons[4], weight=2.0),  # h2 -> o
]

# Add the synapses to the neurons
for synapse in synapses:
    synapse.out_neuron.inputs_synapses.append(synapse)

# Create the neural network
# Layered parameter is required, to define if the network is organized into strict layers
nn = NeuralNetwork(layered=True, neurons=neurons, synapses=synapses)
```
Example code available [here](./examples/xor_neural_network.py)!

#### Execute a neural network
To run your neural network, use the ``forward()`` method by passing the input values as a list of floats. It will return the output values as a list of floats.
```python
result_1 = nn.forward([0.0, 0.0])  # [0.11920292202211755]
result_2 = nn.forward([0.0, 1.0])  # [0.8807970779778823]
result_3 = nn.forward([1.0, 0.0])  # [0.8807970779778823]
result_4 = nn.forward([1.0, 1.0])  # [0.11920292202211755]
```
Example code available [here](./examples/xor_neural_network.py)!

#### Save a neural network
```python
nn.save("my_network.pkl")
```
Example code available [here](./examples/creation_neural_network.py)!

#### Load a neural network
```python
loaded_nn = NeuralNetwork.load("my_network.pkl")
```
Example code available [here](./examples/creation_neural_network.py)!

## NEAT
### DISCLAIMER
> **The algorithm implementation is still under development. Currently, only an __EXTREMELY__ simplified version is available.**

### Explanations
Tona AI provides a very simplified version of the NEAT algorithm ([see disclaimer](#disclaimer)), but it is functional. The principle is simple: you need to define an environment where NEAT will evaluate each individual in the population, retain the top 50%, and for each individual create a mutated copy.

### Usage
#### Setup an environment
The environment is essential as it defines how individuals are evaluated and how their efficiency is calculated.

To create an environment, simply create a class inheriting from the Environment class:
```python
from tona_ai import Environment, Individual

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
```
Example code available [here](./examples/xor_neat.py)!

#### Execute the NEAT algorithm
To run NEAT, you first need to create a population. Start by creating a base **Genome**, which is equivalent to a **Neural Network** with some changes. Then create a population with the ``create()`` method by passing the initial genome. Finally, create a **NEAT** object by passing the population, the environment, and defining the mutation rate and range.
```python
# Create a simple genome with two inputs, one output, and one hidden layer of 4 neurons
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

# Run the NEAT algorithm for 100000 epochs
neat.run(epochs=100000)
```

