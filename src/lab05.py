from helper import DatasetImporterHelper as dih
from helper import DataPreprocessorHelper as dph
from helper import MathHelper as mh


def analyze_iris():
    x, y = dih.load_iris()

    (x_train, y_train), (x_val, y_val) = dph.split_db_2to1(x, y)

    for i in range(3):
        print(f"Class {i} mu:\n{mh.compute_mu(x_train[:, y_train == i])}")
        print(f"Class {i} cv:\n{mh.cv_matrix(x_train[:, y_train == i])}")


def main():
    analyze_iris()


if __name__ == "__main__":
    main()
