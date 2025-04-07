import numpy as np
import pandas as pd
import random as rd

class NeuralNetworkSigmoid:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.weights = [np.random.randn(y, x) * 0.01 for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases = [np.zeros((y, 1)) for y in layer_sizes[1:]]
        self.manual_connections = []  # List of tuples: (from_layer, from_index, to_layer, to_index, weight)
        self.manual_weights = []

    def sigmoid(self, z):
        z = np.clip(z, -50, 50)
        return 1 / (1 + np.exp(-z))


    def sigmoid_derivative(self, a):
        return a * (1 - a)

    def add_manual_connection(self, from_layer, from_index, to_layer, to_index, weight=None):
        if weight is None:
            weight = np.random.randn() * 0.001
        self.manual_connections.append((from_layer, from_index, to_layer, to_index))
        self.manual_weights.append(weight)

    def generate_random_manual_connections(self, num_connections):
        for _ in range(num_connections):
            from_layer = rd.randint(0, len(self.layer_sizes) - 2)
            to_layer = rd.randint(from_layer + 1, len(self.layer_sizes) - 1)
            from_index = rd.randint(0, self.layer_sizes[from_layer] - 1)
            to_index = rd.randint(0, self.layer_sizes[to_layer] - 1)
            print(from_layer,from_index,to_layer,to_index)
            self.add_manual_connection(from_layer, from_index, to_layer, to_index)

    def forward(self, x):
        activations = [x]
        zs = []
        for idx, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = w @ activations[-1] + b
            zs.append(z)
            activations.append(self.sigmoid(z))

        # Apply manual connections
        for i, (from_l, from_i, to_l, to_i) in enumerate(self.manual_connections):
            if from_i < self.layer_sizes[from_l] and to_i < self.layer_sizes[to_l]:
                w = self.manual_weights[i]
                value = activations[from_l][from_i] * w
                value = np.clip(value, -10, 10)
                activations[to_l][to_i] += value


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
            if to_l == len(activations) - 1 and to_i < activations[to_l].shape[0]:
                error = (activations[to_l] - y)
                grad = error[to_i] * self.sigmoid_derivative(activations[to_l][to_i]) * activations[from_l][from_i]
                grads_manual[i] = grad

        return grads_w, grads_b, grads_manual

    def update_params(self, grads_w, grads_b, grads_manual, lr):
        for i in range(len(self.weights)):
            self.weights[i] -= lr * grads_w[i]
            self.biases[i] -= lr * grads_b[i]
        for i in range(len(self.manual_weights)):
            self.manual_weights[i] -= lr * grads_manual[i]

    def train(self, X, Y, epochs=10, lr=0.01,lr_decay=1, batch_size=32):
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
                    # Cross-entropy loss
                    y_pred = self.forward(x)[0][-1]
                    y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)  # éviter log(0)
                    loss = -np.sum(y * np.log(y_pred))
                    total_loss += loss
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
    



class NeuralNetworkSoftmax:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.weights = [np.random.randn(y, x) * 0.01 for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases = [np.zeros((y, 1)) for y in layer_sizes[1:]]
        self.manual_connections = []  # List of tuples: (from_layer, from_index, to_layer, to_index, weight)
        self.manual_weights = []

    def sigmoid(self, z):
        z = np.clip(z, -50, 50)
        return 1 / (1 + np.exp(-z))

    def softmax(self, z):
        e_z = np.exp(z - np.max(z))
        return e_z / np.sum(e_z, axis=0, keepdims=True)


    def sigmoid_derivative(self, a):
        return a * (1 - a)

    def add_manual_connection(self, from_layer, from_index, to_layer, to_index, weight=None):
        if weight is None:
            weight = np.random.randn() * 0.001
        self.manual_connections.append((from_layer, from_index, to_layer, to_index))
        self.manual_weights.append(weight)

    def generate_random_manual_connections(self, num_connections):
        for _ in range(num_connections):
            from_layer = rd.randint(0, len(self.layer_sizes) - 2)
            to_layer = rd.randint(from_layer + 1, len(self.layer_sizes) - 1)
            from_index = rd.randint(0, self.layer_sizes[from_layer] - 1)
            to_index = rd.randint(0, self.layer_sizes[to_layer] - 1)
            print(from_layer,from_index,to_layer,to_index)
            self.add_manual_connection(from_layer, from_index, to_layer, to_index)

    def forward(self, x):
        activations = [x]
        zs = []
        for idx, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = w @ activations[-1] + b
            zs.append(z)
            if idx == len(self.weights) - 1:
                activations.append(self.softmax(z))  # Dernière couche
            else:
                activations.append(self.sigmoid(z))


        # Apply manual connections
        for i, (from_l, from_i, to_l, to_i) in enumerate(self.manual_connections):
            if from_i < self.layer_sizes[from_l] and to_i < self.layer_sizes[to_l]:
                w = self.manual_weights[i]
                value = activations[from_l][from_i] * w
                value = np.clip(value, -10, 10)
                activations[to_l][to_i] += value


        return activations, zs

    def backpropagate(self, x, y):
        activations, zs = self.forward(x)
        grads_w = [np.zeros_like(w) for w in self.weights]
        grads_b = [np.zeros_like(b) for b in self.biases]
        grads_manual = [0 for _ in self.manual_weights]

        # Dernière couche : softmax + cross-entropy
        delta = activations[-1] - y
        grads_w[-1] = delta @ activations[-2].T
        grads_b[-1] = delta

        for l in range(2, len(self.layer_sizes)):
            z = zs[-l]
            sp = self.sigmoid_derivative(activations[-l])
            delta = (self.weights[-l + 1].T @ delta) * sp
            grads_w[-l] = delta @ activations[-l - 1].T
            grads_b[-l] = delta

        # Gradients pour les connexions manuelles
        for i, (from_l, from_i, to_l, to_i) in enumerate(self.manual_connections):
            if to_l == len(activations) - 1 and to_i < activations[to_l].shape[0]:
                error = (activations[to_l] - y)
                grad = error[to_i] * activations[from_l][from_i]
                grads_manual[i] = grad

        return grads_w, grads_b, grads_manual

    def update_params(self, grads_w, grads_b, grads_manual, lr):
        for i in range(len(self.weights)):
            self.weights[i] -= lr * grads_w[i]
            self.biases[i] -= lr * grads_b[i]
        for i in range(len(self.manual_weights)):
            self.manual_weights[i] -= lr * grads_manual[i]

    def train(self, X, Y, epochs=10, lr=0.01,lr_decay=1, batch_size=32):
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
                    # Cross-entropy loss
                    y_pred = self.forward(x)[0][-1]
                    y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)  # éviter log(0)
                    loss = -np.sum(y * np.log(y_pred))
                    total_loss += loss
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
    

