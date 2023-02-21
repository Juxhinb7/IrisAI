def load_iris():
    labels_dict = {
        "Iris-setosa": [0.99, 0.1, 0.1],
        "Iris-versicolor": [0.1, 0.99, 0.1],
        "Iris-virginica": [0.1, 0.1, 0.99]
    }
    f_vector = []
    targets = []
    with open("iris_dataset.txt") as file:
        read_file = file.readlines()
        for rf in read_file:
            vector = rf.strip().split(",")
            features = [float(x) for x in vector[:4]]
            label = vector[4]
            f_vector.append(features)
            targets.append(labels_dict[label])
    return f_vector, targets
