from ann import NeuralNetwork
from load_iris import load_iris

inputs, targets = load_iris()

nn = NeuralNetwork(4, 4, 3, 0.1)

epoch = 100

for i in range(epoch):
    for j in range(len(inputs)):
        nn.train(inputs[j], targets[j])

print(nn.query([7.5, 3.8, 6.6, 2.5]))


