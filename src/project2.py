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
        lda.predict_custom_dir(U=component, x=D) for component in six_main_components
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
    (x_train, y_train), (x_val, y_val) = dh.split_db_2to1(D, L, seed=0)

    lda = LDA(m=1)
    lda.fit(x_train, y_train)

    lda.validate(x_val, y_val=y_val, show_results=True)

    lda.optimize_threshold(steps=10)
    lda.validate(x_val, y_val=y_val, show_results=True)


def classify_pca_preprocess():
    D, L = dh.load_txt("datasets/trainData.txt", features_type=float)
    (x_train, y_train), (x_val, y_val) = dh.split_db_2to1(D, L, seed=0)

    best_config = {"pca_m": 0, "lda_m": 0, "error_rate": 1.0}

    for i in range(2, 6):
        pca = PCA(m=i)
        pca.fit(x_train, y_train)
        x_train_pca = pca.transform(x_train)
        x_val_pca = pca.transform(x_val)
        lda = LDA(m=1)
        lda.fit(x_train_pca, y_train)
        # lda.optimize_threshold(steps=1000)
        print(f"PCA with m = {i} components")
        _, error_rate = lda.validate(x_val_pca, y_val=y_val)
        if error_rate < best_config["error_rate"]:
            best_config["pca_m"] = i
            best_config["lda_m"] = 1
            best_config["error_rate"] = error_rate

    best_config["error_rate"] = str(best_config["error_rate"] * 100) + "%"
    print("\nFinished Loop")
    print(best_config)


def main():
    # analyze_pca_features()
    # analyze_lda_features()
    classify()
    # classify_pca_preprocess()


if __name__ == "__main__":
    main()
