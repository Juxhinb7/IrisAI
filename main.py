from ann import NeuralNetwork
from load_iris import load_irisdataset

# load the training data of inputs and targets
inputs, targets = load_irisdataset(mode="training")

# initialize the neural network with input, hidden, output layers and learning rate
nn = NeuralNetwork(4, 4, 3, 0.1)

# training time
epoch = 100

for i in range(epoch):
    for j in range(len(inputs)):
        nn.train(inputs[j], targets[j])


# load the testing data of inputs and targets
test_inputs, test_targets = load_irisdataset(mode="testing")


# calculate the accuracy of the model
def test(x_test, y_test):
    normalised_pred_targets = []
    for k in range(len(x_test)):
        pred_y = nn.query(x_test[k])
        max_val = max(pred_y)
        normalised_item = []
        for n in pred_y:
            if n != max_val:
                normalised_item.append(0)
            else:
                normalised_item.append(1)

        normalised_pred_targets.append(normalised_item)

    truevalues = 0

    for index in range(len(normalised_pred_targets)):
        if normalised_pred_targets[index] == y_test[index]:
            truevalues += 1

    accuracy = truevalues / len(normalised_pred_targets)
    return accuracy


# predict our new input
print(nn.query([6.2, 3.3, 6.1, 2.4]))

# output the accuracy of the model
print("Model Accuracy:", round(test(test_inputs, test_targets), 2))
