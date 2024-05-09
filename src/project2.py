# from LinearDiscriminantAnalysis import LDA
from PrincipalComponentAnalysis import PCA
from helper import DataHelper as dh
from helper import TimeHelper as th


def analyze_pca_features():


    D, L = dh.load_txt(
        "datasets/trainData.txt",
        features_type=float
    )

    pca = PCA(m=1)

    pca.set_train_data(D, L)
    pca.fit()
    print(pca.get_projected_matrix().shape)


def main():
    analyze_pca_features()



if __name__ == "__main__":
    main()
