def load_irisdataset(mode="training"):
    labels_dict = {
        "Iris-setosa": [1, 0, 0],
        "Iris-versicolor": [0, 1, 0],
        "Iris-virginica": [0, 0, 1]
    }
    f_vector = []
    targets = []
    if mode == "training":
        with open("iris_dataset.txt") as file:
            read_file = file.readlines()
            for rf in read_file:
                vector = rf.strip().split(",")
                features = [float(x) for x in vector[:4]]
                label = vector[4]
                f_vector.append(features)
                targets.append(labels_dict[label])
    elif mode == "testing":
        with open("testdataset.txt") as file:
            read_file = file.readlines()
            for rf in read_file:
                vector = rf.strip().split(",")
                features = [float(x) for x in vector[:4]]
                label = vector[4]
                f_vector.append(features)
                targets.append(labels_dict[label])
    return f_vector, targets
