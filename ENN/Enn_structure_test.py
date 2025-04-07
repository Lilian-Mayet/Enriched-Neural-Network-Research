import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from Neuron import Neuron
from NeuralNetwork import VectorizedNeuralNetwork



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
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, )

# Create and train model
nn = VectorizedNeuralNetwork([784, 64, 10])

# Add manual connection from input neuron 100 to output neuron 5
#nn.add_manual_connection(from_layer=0, from_index=100, to_layer=2, to_index=5, weight=0.05)

nn.train(X_train, Y_train, epochs=25, lr=0.15, batch_size=32,lr_decay=0.95)

# Evaluate
evaluate_model(nn, X_test, Y_test)
