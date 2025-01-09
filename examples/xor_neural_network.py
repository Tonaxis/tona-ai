from tona_ai import ActivationFunction, Layer, LayerType, NeuralNetwork, Neuron, Synapse

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

# Use the neural network
result_1 = nn.forward([0.0, 0.0])
result_2 = nn.forward([0.0, 1.0])
result_3 = nn.forward([1.0, 0.0])
result_4 = nn.forward([1.0, 1.0])

# Print the results
print("Results for the XOR gate:")
print(f"Result for 0 and 0: {result_1}, rounded: {round(result_1[0])}, expected: 0 {"OK" if round(result_1[0]) == 0 else "NOT OK"}")
print(f"Result for 0 and 1: {result_2}, rounded: {round(result_2[0])}, expected: 1 {"OK" if round(result_2[0]) == 1 else "NOT OK"}")
print(f"Result for 1 and 0: {result_3}, rounded: {round(result_3[0])}, expected: 1 {"OK" if round(result_3[0]) == 1 else "NOT OK"}")
print(f"Result for 1 and 1: {result_4}, rounded: {round(result_4[0])}, expected: 0 {"OK" if round(result_4[0]) == 0 else "NOT OK"}")
