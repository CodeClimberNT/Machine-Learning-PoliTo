import numpy as np
import pandas as pd


class DataHelper:
    @staticmethod
    def load_iris():
        import sklearn.datasets

        return (
            sklearn.datasets.load_iris()["data"].T,
            sklearn.datasets.load_iris()["target"],
        )

    @staticmethod
    def load_txt(
        file_path: str,
        delimiter: str = ",",
        features_type: type = float,
        labels_type: type = float,
        return_labels_in_data_matrix: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:

        # data = pd.read_csv(file_path, sep=delimiter, header=None).to_numpy(dtype=features_type)

        # Load the array from the text file and convert all but last column to float
        data = np.loadtxt(file_path, delimiter=delimiter, dtype=features_type)
        
        # return np.rot90(data)

        if return_labels_in_data_matrix:
            x = data
        else:
            # Extract features (all but last column)
            x = data[:, :-1]

        # Extract labels (last column)
        y = data[:, -1].astype(labels_type)

        return x, y

    @staticmethod
    def split_db_2to1(D: np.ndarray, L: np.ndarray, seed=0) -> tuple:

        nTrain = int(D.shape[1] * 2.0 / 3.0)
        np.random.seed(seed)

        idx = np.random.permutation(D.shape[1])

        idxTrain = idx[0:nTrain]
        idxTest = idx[nTrain:]

        DTrain = D[:, idxTrain]
        DVal = D[:, idxTest]

        LTrain = L[idxTrain]
        LVal = L[idxTest]

        return (DTrain, LTrain), (DVal, LVal)


class MathHelper:
    @staticmethod
    def v_col(x: np.ndarray) -> np.ndarray:
        return x.reshape((x.size, 1))

    @staticmethod
    def v_row(x: np.ndarray) -> np.ndarray:
        return x.reshape((1, x.size))

    @staticmethod
    def center_matrix(matrix: np.ndarray) -> np.ndarray:
        mu = matrix.mean(1).reshape(-1, 1)

        return matrix - mu

    @staticmethod
    def cv_matrix(matrix: np.ndarray) -> np.ndarray:
        centered_matrix: np.ndarray = MathHelper.center_matrix(matrix)

        return np.dot(centered_matrix, centered_matrix.T) / float(matrix.shape[0])
