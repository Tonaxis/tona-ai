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

nn.save("test_network.pkl")

loaded_nn = NeuralNetwork.load("test_network.pkl")
