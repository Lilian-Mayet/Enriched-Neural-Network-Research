import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class Neuron:
    def __init__(self, layer_index, neuron_index):
        self.layer_index = layer_index
        self.neuron_index = neuron_index
        self.activation = 0.0
        self.incoming = []  # List of tuples (source_neuron, weight)
        self.outgoing = []  # List of tuples (target_neuron, weight)
        self.delta = 0.0  # For backpropagation

    def add_connection(self, target_neuron, weight):
        self.outgoing.append((target_neuron, weight))
        target_neuron.incoming.append((self, weight))

    def activate(self, input_sum):
        self.activation = 1 / (1 + np.exp(-input_sum))
        return self.activation

    def activation_derivative(self):
        return self.activation * (1 - self.activation)


class CustomNeuralNetwork:
    def __init__(self, layer_sizes):
        self.layers = []
        for i, size in enumerate(layer_sizes):
            self.layers.append([Neuron(i, j) for j in range(size)])

        self.forward_links = {}  # key = (from_layer, to_layer), value = list of (i,j,weight)
        self.initialize_standard_connections()

    def initialize_standard_connections(self):
        for l in range(len(self.layers) - 1):
            for i, neuron_a in enumerate(self.layers[l]):
                for j, neuron_b in enumerate(self.layers[l + 1]):
                    weight = np.random.randn() * 0.01
                    neuron_a.add_connection(neuron_b, weight)
                    self.forward_links.setdefault((l, l + 1), []).append((i, j, weight))

    def add_forward_connection(self, from_layer, from_neuron, to_layer, to_neuron, weight):
        neuron_a = self.layers[from_layer][from_neuron]
        neuron_b = self.layers[to_layer][to_neuron]
        neuron_a.add_connection(neuron_b, weight)
        self.forward_links.setdefault((from_layer, to_layer), []).append((from_neuron, to_neuron, weight))

    def forward_pass(self, input_vector):
        for i, val in enumerate(input_vector):
            self.layers[0][i].activation = val

        for l in range(1, len(self.layers)):
            for neuron in self.layers[l]:
                total_input = sum(prev_neuron.activation * weight for prev_neuron, weight in neuron.incoming)
                neuron.activate(total_input)

        return [neuron.activation for neuron in self.layers[-1]]

    def backpropagate(self, target_output):
        for i, neuron in enumerate(self.layers[-1]):
            error = target_output[i] - neuron.activation
            neuron.delta = error * neuron.activation_derivative()

        for l in reversed(range(1, len(self.layers) - 1)):
            for neuron in self.layers[l]:
                downstream_gradient = sum(target.delta * weight for target, weight in neuron.outgoing)
                neuron.delta = downstream_gradient * neuron.activation_derivative()

    def update_weights(self, learning_rate):
        for layer in self.layers:
            for neuron in layer:
                for i, (target_neuron, weight) in enumerate(neuron.outgoing):
                    gradient = neuron.activation * target_neuron.delta
                    neuron.outgoing[i] = (target_neuron, weight + learning_rate * gradient)

                for source_neuron, _ in neuron.incoming:
                    for i, (n, w) in enumerate(source_neuron.outgoing):
                        if n == neuron:
                            source_neuron.outgoing[i] = (n, w)

    def train(self, training_data, epochs, learning_rate):
        for epoch in range(epochs):
            total_loss = 0
            for input_vector, target_output in training_data:
                
                output = self.forward_pass(input_vector)
                loss = sum((t - o) ** 2 for t, o in zip(target_output, output)) / len(output)
                total_loss += loss
                self.backpropagate(target_output)
                self.update_weights(learning_rate)
                print("Weight updated")
            print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

    def predict(self, input_vector):
        output = self.forward_pass(input_vector)
        return np.argmax(output)


def load_mnist_data(path):
    df = pd.read_csv(path)
    X = df.drop('label', axis=1).values.astype(np.float32)
    y = df['label'].values
    X /= 1.0
    Y = np.zeros((y.size, 10))
    Y[np.arange(y.size), y] = 1
    return X, Y


def evaluate_model(model, test_data):
    correct = 0
    for input_vector, target_output in test_data:
        prediction = model.predict(input_vector)
        if prediction == np.argmax(target_output):
            correct += 1
    accuracy = correct / len(test_data)
    print(f"Accuracy: {accuracy:.2f}")


# Chargement des données MNIST depuis train.csv uniquement
X, Y = load_mnist_data("dataset/train.csv")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

train_data = list(zip(X_train, Y_train))
test_data = list(zip(X_test, Y_test))

# Création du réseau
nn = CustomNeuralNetwork([784,258,128, 64, 10])

# Exemple de connexion longue portée supplémentaire
#nn.add_forward_connection(0, 10, 2, 5, weight=0.05)

# Entraînement
nn.train(train_data[:1000], epochs=5, learning_rate=0.01)

# Évaluation
evaluate_model(nn, test_data[:500])
