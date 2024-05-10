from LinearDiscriminantAnalysis import LDA
from PrincipalComponentAnalysis import PCA
from helper import DataHelper as dh

from Visualizer import Visualizer as vis


def analyze_pca_features():

    D, L = dh.load_txt("datasets/trainData.txt", features_type=float)

    pca = PCA(m=1)

    pca.fit(D, L)
    six_main_components = [pca.take_n_components(i) for i in range(6)]
    six_main_projection = [
        pca.predict_custom_dir(U=component, D=D) for component in six_main_components
    ]
    print(six_main_components[0].shape)
    print(six_main_projection[0].shape)
    print(six_main_projection[0])

    labels_name = {0: "Fake", 1: "Genuine"}
    for i, data in enumerate(six_main_projection):
        vis.plot_hist(
            data,
            L,
            labels_name=labels_name,
            title=f"PCA Feature {i+1}",
            x_label="Label Distribution",
            y_label="Frequency",
            invert_x_axis=True,
        )


def analyze_lda_features():
    D, L = dh.load_txt("datasets/trainData.txt", features_type=float)

    lda = LDA(m=1)

    lda.set_train_data(D, L)
    lda.fit()
    six_main_components = [lda.take_n_components(i) for i in range(6)]
    six_main_projection = [
        lda.predict_custom_dir(U=component, D=D) for component in six_main_components
    ]
    print(six_main_components[0].shape)
    print(six_main_projection[0].shape)
    print(six_main_projection[0])

    labels_name = {0: "Fake", 1: "Genuine"}
    for i, data in enumerate(six_main_projection):
        vis.plot_hist(
            data,
            L,
            labels_name=labels_name,
            title=f"LDA Feature {i+1}",
            x_label="Label Distribution",
            y_label="Frequency",
            invert_x_axis=True,
        )


def classify():
    D, L = dh.load_txt("datasets/trainData.txt", features_type=float)
    (DTrain, LTrain), (DVal, Lval) = dh.split_db_2to1(D, L, seed=0)

    lda = LDA(m=1)
    lda.fit(DTrain, LTrain)

    lda.predict(DVal, show_error_rate=True, LVal=Lval)


def main():
    analyze_pca_features()
    # analyze_lda_features()
    # classify()


if __name__ == "__main__":
    main()
