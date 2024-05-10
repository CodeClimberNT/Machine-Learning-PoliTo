import numpy as np
from helper import DataHelper as d


from LinearDiscriminantAnalysis import LDA
from PrincipalComponentAnalysis import PCA
from Visualizer import Visualizer as vis


# CLASSIFICATION


def classification_no_preprocess() -> None:

    DIris, LIris = d.load_iris()

    D = DIris[:, LIris != 0]

    L = LIris[LIris != 0]

    (DTrain, LTrain), (DVal, LVal) = d.split_db_2to1(D, L)

    lda = LDA(solver="svd", m=1)
    lda.set_train_data(DTrain, LTrain)
    lda.fit()

    PVal = lda.predict(DVal, LVal=LVal, show_error_rate=True)

    print("Labels:     ", LVal)

    print("Predictions:", PVal)
    print(
        "Number of errors:", (PVal != LVal).sum(), "(out of %d samples)" % (LVal.size)
    )

    print("Error rate: %.1f%%" % ((PVal != LVal).sum() / float(LVal.size) * 100))


def classification() -> None:

    DIris, LIris = d.load_iris()

    D = DIris[:, LIris != 0]
    L = LIris[LIris != 0]
    # Train: training
    # Val: validation
    (DTrain, LTrain), (DVal, LVal) = d.split_db_2to1(D, L)

    pca = PCA(m=2)
    pca.fit(DTrain, LTrain)


    DVal_pca = pca.predict(DVal)

    DTrain_pca = pca.get_projected_matrix()

    lda = LDA(solver="svd", m=1)
    lda.fit(DTrain_pca, LTrain)

    PVal = lda.predict(DVal_pca, LVal=LVal, show_error_rate=True)


    pca_matrix = np.vstack((DTrain_pca, LTrain))
    vis.plot_scatter_matrix(
        pca_matrix,
        LTrain,
        labels_name={1: "Versicolor", 2: "Virginica"},
        title="LDA Projection of Iris Dataset",
        x_label="LDA Projection 1",
        y_label="LDA Projection 2",
        invert_y_axis=True,
        invert_x_axis=True,
    )

    lda_matrix = lda.get_projected_matrix()

    lda_train = np.vstack((lda_matrix, LTrain))

    lda_predict = np.vstack((DVal_pca, PVal))

    vis.plot_hist(
        lda_train,
        LTrain,
        labels_name={1: "Versicolor", 2: "Virginica"},
        title="LDA Training Set",
        x_label="LDA Projection 1",
        y_label="LDA Projection 2",
        # invert_y_axis=True,
        invert_x_axis=True,
    )

    vis.plot_hist(
        lda_predict,
        LVal,
        labels_name={1: "Versicolor", 2: "Virginica"},
        title="LDA Prediction Set",
        x_label="LDA Projection 1",
        y_label="LDA Projection 2",
        # invert_y_axis=True,
        invert_x_axis=True,
    )



def test_lda():
    D, L = d.load_iris()
    lda = LDA(solver="svd", m=2)
    lda.set_train_data(D, L)
    lda.fit()
    y = lda.get_projected_matrix()
    print(y)
    vis.plot_scatter_matrix(
        y,
        L,
        labels_name={0: "Setosa", 1: "Versicolor", 2: "Virginica"},
        title="LDA Projection of Iris Dataset",
        x_label="LDA Projection 1",
        y_label="LDA Projection 2",
        invert_y_axis=True,
        invert_x_axis=True,
    )

    lda.set_dimensions(1)
    lda.fit()
    y = lda.get_projected_matrix()
    vis.plot_hist(
        y,
        L,
        labels_name={0: "Setosa", 1: "Versicolor", 2: "Virginica"},
        title="LDA Projection of Iris Dataset",
        x_label="LDA Projection",
        y_label="Frequency",
        invert_x_axis=True,
    )


def test_pca():
    D, L = d.load_iris()
    pca = PCA(m=2)
    pca.set_train_data(D, L)
    pca.fit()
    y = pca.get_projected_matrix()

    vis.plot_scatter_matrix(
        y,
        L,
        labels_name={0: "Setosa", 1: "Versicolor", 2: "Virginica"},
        title="LDA Projection of Iris Dataset",
        x_label="LDA Projection 1",
        y_label="LDA Projection 2",
        invert_y_axis=True,
        # invert_x_axis=True,
    )

    pca.set_dimensions(1)
    pca.fit()
    y = pca.get_projected_matrix()
    print(y)
    # apply label to the last column
    y = np.vstack((y, L))
    vis.plot_hist(
        y,
        L,
        labels_name={0: "Setosa", 1: "Versicolor", 2: "Virginica"},
        title="PCA Projection of Iris Dataset hist",
        x_label="PCA Projection",
        y_label="Frequency",
        # invert_x_axis=True,
    )


def main() -> None:
    # test_lda()
    # test_pca()
    # classification_no_preprocess()

    classification()


if __name__ == "__main__":

    # Configure logging level

    # logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
    main()
