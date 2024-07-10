from src.models.gaussian_models import GaussianModel
from src.helpers import DatasetImporterHelper as ds, MathHelper as mh

import numpy as np
from matplotlib import pyplot as plt


def dataset_multivariate_analysis():
    x, y = ds.load_train_project()

    unique_features = len(x)
    print(f"Unique features: {unique_features}")
    for i in range(unique_features):
        mvg = GaussianModel().fit(X=mh.v_row(x[i]))
        XPlot = np.linspace(-8, 12, 1000)

        plt.figure()
        plt.hist(x[i].ravel(), bins=50, density=True)
        plt.plot(XPlot.ravel(), mvg.pdf_GAU_ND(mh.v_row(XPlot)))
        plt.title(f"Feature {i + 1}")
        plt.show()


def main():
    dataset_multivariate_analysis()


if __name__ == "__main__":
    main()
