# from LinearDiscriminantAnalysis import LDA
from PrincipalComponentAnalysis import PCA
from helper import DataHelper as dh
import numpy as np
import timeit

# import time


def analyze_pca_features():

    # np.loadtxt("datasets/trainData.txt", delimiter=",")

    start_time = timeit.default_timer()
    D, L = dh.load_txt(
        "datasets/trainData.txt",
        delimiter=",",
        features_type=float,
        labels_type=float,
        return_labels_in_data_matrix=True,
    )
    end_time = timeit.default_timer()
    execution_time = end_time - start_time
    print(f"Load method execution time: {execution_time} seconds")


    D = np.rot90(D, k=1)

    print(type(D))

    print(D.shape)
    print(D[:5])

    pca = PCA(m=1)

    pca.set_train_data(D, L)
    pca.fit()
    print(pca.get_projected_matrix())


def main():
    analyze_pca_features()


if __name__ == "__main__":
    main()
