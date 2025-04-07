import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class VectorizedNeuralNetwork:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.weights = [np.random.randn(y, x) * 0.01 for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases = [np.zeros((y, 1)) for y in layer_sizes[1:]]
        self.manual_connections = []  # List of tuples: (from_layer, from_index, to_layer, to_index, weight)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, a):
        return a * (1 - a)

    def add_manual_connection(self, from_layer, from_index, to_layer, to_index, weight=None):
        if weight is None:
            weight = np.random.randn() * 0.01
        self.manual_connections.append((from_layer, from_index, to_layer, to_index, weight))

    def forward(self, x):
        activations = [x]
        zs = []
        for idx, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = w @ activations[-1] + b
            zs.append(z)
            activations.append(self.sigmoid(z))

        # Apply manual connections
        for from_l, from_i, to_l, to_i, w in self.manual_connections:
            if from_l < len(activations) and to_l < len(activations):
                activations[to_l][to_i] += activations[from_l][from_i] * w

        return activations, zs

    def backpropagate(self, x, y):
        activations, zs = self.forward(x)
        grads_w = [np.zeros_like(w) for w in self.weights]
        grads_b = [np.zeros_like(b) for b in self.biases]

        delta = (activations[-1] - y) * self.sigmoid_derivative(activations[-1])
        grads_w[-1] = delta @ activations[-2].T
        grads_b[-1] = delta

        for l in range(2, len(self.layer_sizes)):
            delta = (self.weights[-l + 1].T @ delta) * self.sigmoid_derivative(activations[-l])
            grads_w[-l] = delta @ activations[-l - 1].T
            grads_b[-l] = delta

        return grads_w, grads_b

    def update_params(self, grads_w, grads_b, lr):
        for i in range(len(self.weights)):
            self.weights[i] -= lr * grads_w[i]
            self.biases[i] -= lr * grads_b[i]

    def train(self, X, Y, epochs=10, lr=0.01, batch_size=32):
        n = X.shape[0]
        for epoch in range(epochs):
            perm = np.random.permutation(n)
            X_shuffled = X[perm]
            Y_shuffled = Y[perm]
            total_loss = 0
            for i in range(0, n, batch_size):
                x_batch = X_shuffled[i:i+batch_size].T
                y_batch = Y_shuffled[i:i+batch_size].T
                grads_w_sum = [np.zeros_like(w) for w in self.weights]
                grads_b_sum = [np.zeros_like(b) for b in self.biases]
                for j in range(x_batch.shape[1]):
                    x = x_batch[:, j:j+1]
                    y = y_batch[:, j:j+1]
                    grads_w, grads_b = self.backpropagate(x, y)
                    grads_w_sum = [gw + dw for gw, dw in zip(grads_w_sum, grads_w)]
                    grads_b_sum = [gb + db for gb, db in zip(grads_b_sum, grads_b)]
                    total_loss += np.mean((self.forward(x)[0][-1] - y)**2)
                self.update_params([g / batch_size for g in grads_w_sum],
                                   [g / batch_size for g in grads_b_sum], lr)
            print(f"Epoch {epoch+1}, Loss: {total_loss / (n/batch_size):.4f}")

    def predict(self, X):
        A, _ = self.forward(X.T)
        predictions = np.argmax(A[-1], axis=0)
        return predictions


def load_mnist_data(path):
    df = pd.read_csv(path)
    X = df.drop('label', axis=1).values.astype(np.float32)
    y = df['label'].values
    X /= 1.0
    Y = np.zeros((y.size, 10))
    Y[np.arange(y.size), y] = 1
    return X, Y


def evaluate_model(model, X_test, Y_test):
    predictions = model.predict(X_test)
    correct = np.sum(predictions == np.argmax(Y_test, axis=1))
    accuracy = correct / len(Y_test)
    print(f"Accuracy: {accuracy:.2f}")


# Load and split data
X, Y = load_mnist_data("dataset/train.csv")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Create and train model
nn = VectorizedNeuralNetwork([784, 64, 10])

# Add manual connection from input neuron 100 to output neuron 5
#nn.add_manual_connection(from_layer=0, from_index=100, to_layer=2, to_index=5, weight=0.05)

nn.train(X_train, Y_train, epochs=10, lr=0.1, batch_size=32)

# Evaluate
evaluate_model(nn, X_test, Y_test)
