import numpy as np


def save_data_with_numpy(file, X_train, y_train, X_test, y_test, compressed=True):
    """_summary_

    Args:
        file (_type_): _description_
        X_train (_type_): _description_
        y_train (_type_): _description_
        X_test (_type_): _description_
        y_test (_type_): _description_
        compressed (bool, optional): _description_. Defaults to True.

    Example usage:
        file = "mnist.npz" # file needs ".npz"

        # if loading is important
        mnist = np.load('mnist_scaled.npz')
        mnist.files

        X_train, y_train, X_test, y_test = [mnist[f] for f in ['X_train', 'y_train',
                                            'X_test', 'y_test']]
    """
    if compressed:
        np.savez_compressed(
            file, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
        )
    else:
        np.savez(file, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)


def load_data_with_numpy(file):
    """_summary_

    Args:
        file (_type_): _description_
    """
    data = np.load(file)
    return data
