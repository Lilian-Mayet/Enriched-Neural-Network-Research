import numpy as np

class Neuron:
    def __init__(self, layer_index, neuron_index):
        self.layer_index = layer_index
        self.neuron_index = neuron_index
        self.activation = 0.0
        self.incoming = []  # List of tuples (source_neuron, weight)
        self.outgoing = []  # List of tuples (target_neuron, weight)

    def add_connection(self, target_neuron, weight):
        self.outgoing.append((target_neuron, weight))
        target_neuron.incoming.append((self, weight))

    def activate(self, input_sum):
        self.activation = 1 / (1 + np.exp(-input_sum))
        return self.activation


class CustomNeuralNetwork:
    def __init__(self, layer_sizes):
        self.layers = []
        for i, size in enumerate(layer_sizes):
            self.layers.append([Neuron(i, j) for j in range(size)])

        self.forward_links = {}  # key = (from_layer, to_layer), value = list of (i,j,weight)

    def add_forward_connection(self, from_layer, from_neuron, to_layer, to_neuron, weight):
        neuron_a = self.layers[from_layer][from_neuron]
        neuron_b = self.layers[to_layer][to_neuron]
        neuron_a.add_connection(neuron_b, weight)
        self.forward_links.setdefault((from_layer, to_layer), []).append((from_neuron, to_neuron, weight))

    def forward_pass(self, input_vector):
        # Set activations of input layer
        for i, val in enumerate(input_vector):
            self.layers[0][i].activation = val

        # Go layer by layer
        for l in range(1, len(self.layers)):
            for neuron in self.layers[l]:
                total_input = sum(prev_neuron.activation * weight for prev_neuron, weight in neuron.incoming)
                neuron.activate(total_input)

        # Return activations of output layer
        return [neuron.activation for neuron in self.layers[-1]]

    def backpropagate(self, target_output):
        # Implement a custom backpropagation algorithm
        pass

    def update_weights(self, learning_rate):
        # Implement weight update logic, possibly differentiating between classic and long-range connections
        pass

    def train(self, training_data, epochs, learning_rate):
        for epoch in range(epochs):
            for input_vector, target_output in training_data:
                self.forward_pass(input_vector)
                self.backpropagate(target_output)
                self.update_weights(learning_rate)


# Example of use:
# Create a network with 3 layers: input(3), hidden(5), output(2)
nn = CustomNeuralNetwork([3, 5, 2])

# Add long-range connection from input layer to output layer
nn.add_forward_connection(0, 1, 2, 0, weight=0.1)  # from neuron 1 of layer 0 to neuron 0 of layer 2

# Placeholder for training data
data = [
    ([0.5, 0.3, 0.9], [1, 0]),
    ([0.1, 0.4, 0.6], [0, 1])
]

# Train the network
nn.train(data, epochs=10, learning_rate=0.01)
