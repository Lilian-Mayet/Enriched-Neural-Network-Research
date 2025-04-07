import numpy as np
import pandas as pd

class VectorizedNeuralNetwork:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.weights = [np.random.randn(y, x) * 0.01 for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases = [np.zeros((y, 1)) for y in layer_sizes[1:]]
        self.manual_connections = []  # List of tuples: (from_layer, from_index, to_layer, to_index, weight)
        self.manual_weights = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, a):
        return a * (1 - a)

    def add_manual_connection(self, from_layer, from_index, to_layer, to_index, weight=None):
        if weight is None:
            weight = np.random.randn() * 0.01
        self.manual_connections.append((from_layer, from_index, to_layer, to_index))
        self.manual_weights.append(weight)

    def forward(self, x):
        activations = [x]
        zs = []
        for idx, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = w @ activations[-1] + b
            zs.append(z)
            activations.append(self.sigmoid(z))

        # Apply manual connections
        for i, (from_l, from_i, to_l, to_i) in enumerate(self.manual_connections):
            w = self.manual_weights[i]
            activations[to_l][to_i] += activations[from_l][from_i] * w

        return activations, zs

    def backpropagate(self, x, y):
        activations, zs = self.forward(x)
        grads_w = [np.zeros_like(w) for w in self.weights]
        grads_b = [np.zeros_like(b) for b in self.biases]
        grads_manual = [0 for _ in self.manual_weights]

        delta = (activations[-1] - y) * self.sigmoid_derivative(activations[-1])
        grads_w[-1] = delta @ activations[-2].T
        grads_b[-1] = delta

        for l in range(2, len(self.layer_sizes)):
            delta = (self.weights[-l + 1].T @ delta) * self.sigmoid_derivative(activations[-l])
            grads_w[-l] = delta @ activations[-l - 1].T
            grads_b[-l] = delta

        # Gradients for manual connections
        for i, (from_l, from_i, to_l, to_i) in enumerate(self.manual_connections):
            error = (activations[-1] - y)
            grad = error[to_i] * self.sigmoid_derivative(activations[to_l][to_i]) * activations[from_l][from_i]
            grads_manual[i] = grad

        return grads_w, grads_b, grads_manual

    def update_params(self, grads_w, grads_b, grads_manual, lr):
        for i in range(len(self.weights)):
            self.weights[i] -= lr * grads_w[i]
            self.biases[i] -= lr * grads_b[i]
        for i in range(len(self.manual_weights)):
            self.manual_weights[i] -= lr * grads_manual[i]

    def train(self, X, Y, epochs=10, lr=0.01, batch_size=32,lr_decay=1):
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
                grads_manual_sum = [0 for _ in self.manual_weights]
                for j in range(x_batch.shape[1]):
                    x = x_batch[:, j:j+1]
                    y = y_batch[:, j:j+1]
                    grads_w, grads_b, grads_manual = self.backpropagate(x, y)
                    grads_w_sum = [gw + dw for gw, dw in zip(grads_w_sum, grads_w)]
                    grads_b_sum = [gb + db for gb, db in zip(grads_b_sum, grads_b)]
                    grads_manual_sum = [gm + dm for gm, dm in zip(grads_manual_sum, grads_manual)]
                    total_loss += np.mean((self.forward(x)[0][-1] - y)**2)
                self.update_params(
                    [g / batch_size for g in grads_w_sum],
                    [g / batch_size for g in grads_b_sum],
                    [g / batch_size for g in grads_manual_sum], lr)
            print(f"Epoch {epoch+1}, Loss: {total_loss / (n/batch_size):.4f}")
            lr = lr*lr_decay

    def predict(self, X):
        A, _ = self.forward(X.T)
        predictions = np.argmax(A[-1], axis=0)
        return predictions
