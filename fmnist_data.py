import numpy as np


def load_fashion_mnist_data(no_training_samples, no_testing_samples, random_seed=0):
    np.random.seed(random_seed)

    data = np.load("data/fashion_mnist.npz")

    index_train = np.arange(data["X_train"].shape[0])
    np.random.shuffle(index_train)

    index_test = np.arange(data["X_test"].shape[0])
    np.random.shuffle(index_test)

    x_train = data["X_train"][index_train[0:no_training_samples], :]
    y_train = data["Y_train"][index_train[0:no_training_samples], :]
    x_test = data["X_test"][index_test[0:no_testing_samples], :]
    y_test = data["Y_test"][index_test[0:no_testing_samples], :]

    # normalize in 0..1
    x_train = x_train.astype('float64') / 255.
    x_test = x_test.astype('float64') / 255.

    return x_train, y_train, x_test, y_test
