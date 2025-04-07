class Neuron:
    def __init__(self):
        self.incoming = []  # List of tuples: (source_neuron, weight)
        self.outgoing = []  # List of tuples: (target_neuron, weight)
        self.activation = 0
        self.delta = 0