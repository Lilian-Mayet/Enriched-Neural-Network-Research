import numpy as np
import pandas as pd

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
        for i, val in enumerate(input_vector):
            self.layers[0][i].activation = val

        for l in range(1, len(self.layers)):
            for neuron in self.layers[l]:
                total_input = sum(prev_neuron.activation * weight for prev_neuron, weight in neuron.incoming)
                neuron.activate(total_input)

        return [neuron.activation for neuron in self.layers[-1]]

    def backpropagate(self, target_output):
        pass  # À implémenter

    def update_weights(self, learning_rate):
        pass  # À implémenter

    def train(self, training_data, epochs, learning_rate):
        for epoch in range(epochs):
            for input_vector, target_output in training_data:
                self.forward_pass(input_vector)
                self.backpropagate(target_output)
                self.update_weights(learning_rate)

    def predict(self, input_vector):
        output = self.forward_pass(input_vector)
        return np.argmax(output)


def load_mnist_data(path):
    df = pd.read_csv(path)
    X = df.drop('label', axis=1).values.astype(np.float32)
    y = df['label'].values

    X /= 1.0  # Les pixels sont déjà entre 0 et 1

    # One-hot encoding
    Y = np.zeros((y.size, 10))
    Y[np.arange(y.size), y] = 1

    return list(zip(X, Y))


def evaluate_model(model, test_data):
    correct = 0
    for input_vector, target_output in test_data:
        prediction = model.predict(input_vector)
        if prediction == np.argmax(target_output):
            correct += 1
    accuracy = correct / len(test_data)
    print(f"Accuracy: {accuracy:.2f}")


# Chargement des données MNIST
train_data = load_mnist_data("dataset/train.csv")
test_data = load_mnist_data("dataset/test.csv")

# Création du réseau : 784 entrées (pixels), 64 neurones cachés, 10 sorties (chiffres)
nn = CustomNeuralNetwork([784,258,128, 64, 10])

# Exemple de connexion longue portée
nn.add_forward_connection(0, 10, 2, 5, weight=0.05)  # de l'entrée 10 à la sortie 5

# Entraînement
nn.train(train_data[:1000], epochs=5, learning_rate=0.01)  # Sous-échantillon pour aller plus vite

# Évaluation
evaluate_model(nn, test_data[:500])
