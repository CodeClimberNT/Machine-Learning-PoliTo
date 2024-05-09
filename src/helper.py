import numpy as np

import timeit


class TimeHelper:
    @staticmethod
    def start_timer():
        return timeit.default_timer()

    def end_timer(start_time, print_time=True) -> float:
        if print_time:
            print(f"Execution time: {timeit.default_timer() - start_time} seconds")
        return timeit.default_timer() - start_time


class DataHelper:
    @staticmethod
    def load_iris():
        import sklearn.datasets

        return (
            sklearn.datasets.load_iris()["data"].T,
            sklearn.datasets.load_iris()["target"],
        )

    # @staticmethod
    # @DeprecationWarning
    # def load_txt(
    #     file_path: str,
    #     delimiter: str = ",",
    #     features_type: type = float,
    #     labels_type: type = float,
    #     return_labels_in_data_matrix: bool = False,
    # ) -> tuple[np.ndarray, np.ndarray]:

    #     # Load the array from the text file and convert all but last column to float
    #     data = np.loadtxt(file_path, delimiter=delimiter, dtype=features_type)

    #     # return np.rot90(data)

    #     if return_labels_in_data_matrix:
    #         x = data
    #     else:
    #         # Extract features (all but last column)
    #         x = data[:, :-1]

    #     # Extract labels (last column)
    #     y = data[:, -1].astype(labels_type)

    #     return x, y

    @staticmethod
    def load_txt(
        file_path: str,
        sep: str = ",",
        labels_need_map: bool = False,
        labels_map: dict[str | int, int] = None,
        features_type: type = float,
        labels_type: type = np.int32,
    ) -> tuple[np.ndarray, np.ndarray]:
        data_list: list = []
        labels_list: list = []

        if labels_map is None and labels_need_map:
            print("Using default labels map for Iris dataset")
            labels_map: dict[str | int, int] = {
                "Iris-setosa": 0,
                "Iris-versicolor": 1,
                "Iris-virginica": 2,
            }

        with open(file_path) as f:
            for line in f:
                try:
                    attrs = line.split(sep)[0:-1]
                    attrs = MathHelper.v_col(
                        np.array([features_type(i) for i in attrs])
                    )
                    data_list.append(attrs)

                    label = line.split(sep)[-1].strip()
                    
                    if labels_need_map:
                        label = labels_map[label]

                    labels_list.append(label)

                except FileNotFoundError as e:
                    print(f"File not found: {e}")

        return np.hstack(data_list, dtype=features_type), np.array(
            labels_list, dtype=labels_type
        )

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
