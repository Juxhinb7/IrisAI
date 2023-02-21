import numpy as np

from activation_functions.sigmoid import sigmoid


class NeuralNetwork:

    # initialize the neural network
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # set the number of nodes in each layer
        self.input_nodes: int = input_nodes
        self.hidden_nodes: int = hidden_nodes
        self.output_nodes: int = output_nodes

        # initialize random weights
        self.wih = (np.random.rand(self.hidden_nodes, self.input_nodes) - 0.5)
        self.who = (np.random.rand(self.output_nodes, self.hidden_nodes) - 0.5)

        # learning rate
        self.lr = learning_rate

    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        x_hidden = np.dot(self.wih, inputs)
        o_hidden = []
        for item in x_hidden:
            for x in item:
                o_hidden.append(sigmoid(x))
        o_hidden = np.array(o_hidden, ndmin=2).T
        x_output = np.dot(self.who, o_hidden)
        o_output = []
        for item in x_output:
            for x in item:
                o_output.append(sigmoid(x))
        o_output = np.array(o_output, ndmin=2).T
        output_errors = targets - o_output
        hidden_errors = np.dot(self.who.T, output_errors)
        self.who += self.lr * np.dot((output_errors * o_output * (1 - o_output)), np.transpose(o_hidden))
        self.wih += self.lr * np.dot((hidden_errors * o_hidden * (1 - o_hidden)), np.transpose(inputs))

    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T
        x_hidden = np.dot(self.wih, inputs)
        o_hidden = []
        for item in x_hidden:
            for x in item:
                o_hidden.append(sigmoid(x))
        o_hidden = np.array(o_hidden, ndmin=2).T
        x_output = np.dot(self.who, o_hidden)
        o_output = []
        for item in x_output:
            for x in item:
                o_output.append(sigmoid(x))
        o_output = np.array(o_output, ndmin=2).T
        return o_output
