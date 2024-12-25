import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import KernelPCA


def test_linear_seperaty_kpca(X, y, kernel, gamma, n_components):
    k_pca = KernelPCA(
        n_components=n_components, kernel=kernel, gamma=gamma
    ).fit_transform(X)

    if n_components == 2:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))
        ax[0].scatter(
            k_pca[y == 0, 0], k_pca[y == 0, 1], color="red", marker="^", alpha=0.5
        )
        ax[0].scatter(
            k_pca[y == 1, 0], k_pca[y == 1, 1], color="blue", marker="o", alpha=0.5
        )
        ax[1].scatter(
            k_pca[y == 0, 0],
            np.zeros((50, 1)) + 0.02,
            color="red",
            marker="^",
            alpha=0.5,
        )
        ax[1].scatter(
            k_pca[y == 1, 0],
            np.zeros((50, 1)) - 0.02,
            color="blue",
            marker="o",
            alpha=0.5,
        )
        ax[0].set_xlabel("PC1")
        ax[0].set_ylabel("PC2")
        ax[1].set_ylim([-1, 1])
        ax[1].set_yticks([])
        ax[1].set_xlabel("PC1")
        plt.tight_layout()
        plt.show()
    else:
        print(f"Only works if n_components=2, given n_components={n_components}")
