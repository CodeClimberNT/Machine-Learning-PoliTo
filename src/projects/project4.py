import numpy as np
from matplotlib import pyplot as plt

from src.helpers import (
    DatasetImporterHelper as dih,
    MathHelper as mh,
    DataHandler as dh,
)

from src.models.gaussian_models import (
    MultivariateGaussianModel as MVG,
    NaiveBayesBaseGaussianModel as NBG,
    TiedCovarianceBaseGaussianModel as TCG,
)
from src.models import (
    LinearDiscriminantAnalysis as LDA,
    PrincipalComponentAnalysis as PCA,
)


def apply_mvg_model(features_to_remove: list | None = None):
    (x_train, y_train), (x_val, y_val) = dih.load_train_project_splitted()
    if features_to_remove:
        x_train = dh.remove_features(x_train, features_to_remove)
        x_val = dh.remove_features(x_val, features_to_remove)
    mvg_model = MVG().fit(x_train, y_train)
    llr = mvg_model.log_likelihood_ratio(x_val)
    error_rate = mvg_model.compute_error_rate(llr, y_val)
    return error_rate


def apply_tied_model(features_to_remove: list | None = None):
    (x_train, y_train), (x_val, y_val) = dih.load_train_project_splitted()
    if features_to_remove:
        x_train = dh.remove_features(x_train, features_to_remove)
        x_val = dh.remove_features(x_val, features_to_remove)

    tcg_model = TCG().fit(x_train, y_train)
    llr = tcg_model.log_likelihood_ratio(x_val)
    error_rate = tcg_model.compute_error_rate(llr, y_val)
    return error_rate


def apply_mvg_lda_model(features_to_remove: list | None = None):
    (x_train, y_train), (x_val, y_val) = dih.load_train_project_splitted()
    if features_to_remove:
        x_train = dh.remove_features(x_train, features_to_remove)
        x_val = dh.remove_features(x_val, features_to_remove)

    lda = LDA(m=1).fit(x_train, y_train)
    x_train_lda = lda.transform(x_train)
    x_val_lda = lda.transform(x_val)

    mvg = MVG().fit(x_train_lda, y_train)

    llr = mvg.log_likelihood_ratio(x_val_lda)
    error_rate = mvg.compute_error_rate(llr, y_val)
    return error_rate


def apply_naive_model(features_to_remove: list | None = None):
    (x_train, y_train), (x_val, y_val) = dih.load_train_project_splitted()
    if features_to_remove:
        x_train = dh.remove_features(x_train, features_to_remove)
        x_val = dh.remove_features(x_val, features_to_remove)

    nbg_model = NBG().fit(x_train, y_train)
    llr = nbg_model.log_likelihood_ratio(x_val)
    error_rate = nbg_model.compute_error_rate(llr, y_val)
    return error_rate


def compare_models():
    error_rate_mvg = apply_mvg_model()
    error_rate_tcg = apply_tied_model()
    error_rate_mvg_lda = apply_mvg_lda_model()
    error_rate_naive = apply_naive_model()
    print(f"MVG error rate:\t\t {error_rate_mvg:.2f}%")
    print(f"LDA->MVG error rate: {error_rate_mvg_lda:.2f}%")
    print(f"Naive error rate:\t {error_rate_naive:.2f}%")
    print(f"TIED error rate:\t {error_rate_tcg:.2f}%")


def analyze_cov_matrices():
    (x_train, y_train), (x_val, y_val) = dih.load_train_project_splitted()
    mvg_model = MVG().fit(x_train, y_train)
    for c in mvg_model.classes:
        print(f"Class {c}")
        print(f"Cov:\n{mvg_model.h_params[c]['sigma_']}")
        cov = mvg_model.h_params[c]["sigma_"]
        corr1 = mh.pearson_correlation_homemade(cov)
        corr2 = mh.pearson_correlation_numpy(cov)
        print(f"Correlation matrix (homemade):\n{corr1}")
        print(f"Correlation matrix (numpy):\n{corr2}")
        print()


def calculate_goodness():
    (x_train, y_train), (x_val, y_val) = dih.load_train_project_splitted()

    unique_features = len(x_train)
    print(f"Unique features: {unique_features}")
    for i in range(unique_features):
        mvg = MVG().fit(X=mh.v_row(x_train[i]))
        XPlot = np.linspace(-8, 12, 1000)

        plt.figure()
        plt.hist(x_train[i].ravel(), bins=50, density=True)
        plt.plot(XPlot.ravel(), mvg.pdf_GAU_ND(mh.v_row(XPlot)))
        plt.title(f"Feature {i + 1}")
        plt.show()


#     feature 5 and 6 does not have a normal distribution


def compare_models_one_to_four():
    error_rate_mvg = apply_mvg_model([4, 5])
    error_rate_tcg = apply_tied_model([4, 5])
    error_rate_mvg_lda = apply_mvg_lda_model([4, 5])
    error_rate_naive = apply_naive_model([4, 5])
    print(f"MVG error rate:\t\t {error_rate_mvg:.2f}%")
    print(f"LDA->MVG error rate: {error_rate_mvg_lda:.2f}%")
    print(f"Naive error rate:\t {error_rate_naive:.2f}%")
    print(f"TIED error rate:\t {error_rate_tcg:.2f}%")
    return error_rate_mvg, error_rate_tcg, error_rate_mvg_lda, error_rate_naive


def apply_mvg_on_selected_features():
    (x_train, y_train), (x_val, y_val) = dih.load_train_project_splitted()

    features_pairs = [[0, 1], [2, 3]]
    models = [MVG(), TCG()]
    model_names = ["MVG", "TCG"]

    error_rates = {}

    for pair in features_pairs:
        x_train_pair = dh.remove_features(x_train, pair)
        x_val_pair = dh.remove_features(x_val, pair)

        for model, name in zip(models, model_names):
            model.fit(x_train_pair, y_train)
            llr = model.log_likelihood_ratio(x_val_pair)
            predicted_val = np.zeros(x_val_pair.shape[1], dtype=np.int32)
            threshold = 0  # Adjust threshold as needed
            predicted_val[llr >= threshold] = 1
            predicted_val[llr < threshold] = 0

            error_rate = (np.sum(predicted_val != y_val) / y_val.size) * 100
            error_rates[f"{name} Features {pair}"] = error_rate
            print(f"{name} Features {pair}: {error_rate:.2f}%")

    return error_rates


def apply_pca_preprocessing():
    (x_train, y_train), (x_val, y_val) = dih.load_train_project_splitted()
    error_rates = {}
    for i in range(2, 6):
        pca = PCA(m=i)
        x_train_pca = pca.fit_transform(x_train, y_train)
        x_val_pca = pca.transform(x_val)
        # lda = LDA(m=5).fit(x_train_pca, y_train)
        # x_train_pca_lda = lda.transform(x_train_pca)
        # x_val_pca_lda = lda.transform(x_val_pca)
        mvg_model = MVG().fit(x_train_pca, y_train)
        llr = mvg_model.log_likelihood_ratio(x_val_pca)
        error_rate_mvg = mvg_model.compute_error_rate(llr, y_val)
        print(f"PCA - MVG with m = {i} components: {error_rate_mvg:.2f}%")

        nbg_model = NBG().fit(x_train_pca, y_train)
        llr = nbg_model.log_likelihood_ratio(x_val_pca)
        error_rate_naive = nbg_model.compute_error_rate(llr, y_val)
        print(f"PCA - Naive with m = {i} components: {error_rate_naive:.2f}%")

        tcg_model = TCG().fit(x_train_pca, y_train)
        llr = tcg_model.log_likelihood_ratio(x_val_pca)
        error_rate_tied = tcg_model.compute_error_rate(llr, y_val)
        print(f"PCA - Tied with m = {i} components: {error_rate_tied:.2f}%")
        print()
        error_rates[i] = {
            "mvg_error_rate": error_rate_mvg,
            "naive_error_rate": error_rate_naive,
            "tied_error_rate": error_rate_tied,
        }

    return error_rates


def main():
    compare_models()
    print()
    analyze_cov_matrices()
    print()
    # calculate_goodness()
    print()
    compare_models_one_to_four()
    print()
    apply_mvg_on_selected_features()
    print()
    apply_pca_preprocessing()
    print()


if __name__ == "__main__":
    main()
