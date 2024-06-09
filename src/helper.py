import numpy as np
import os
import timeit


class DatasetImporterHelper:
    @staticmethod
    def load_iris():
        import sklearn.datasets

        return (
            sklearn.datasets.load_iris()["data"].T,
            sklearn.datasets.load_iris()["target"],
        )

    @staticmethod
    def load_train_project():
        abs_path: str = "datasets/project/trainData.txt"
        rel_path: str = "../datasets/project/trainData.txt"
        if DatasetImporterHelper.file_exists(abs_path):
            return DatasetImporterHelper.load_txt(
                abs_path, features_type=float
            )
        elif DatasetImporterHelper.file_exists(rel_path):
            return DatasetImporterHelper.load_txt(
                rel_path, features_type=float
            )
        else:
            print("File not found")
            return None

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
    @DeprecationWarning
    def load_txt_with_np(
            file_path: str,
            delimiter: str = ",",
            features_type: type = float,
            labels_type: type = float,
            return_labels_in_data_matrix: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:

        # Load the array from the text file and convert all but last column to float
        data = np.loadtxt(file_path, delimiter=delimiter, dtype=features_type)

        if return_labels_in_data_matrix:
            x = data
        else:
            # Extract features (all but last column)
            x = data[:, :-1]

        # Extract labels (last column)
        y = data[:, -1].astype(labels_type)

        return x, y

    @staticmethod
    def file_exists(file_path):
        return os.path.isfile(file_path)


class DataPreprocessorHelper:
    @staticmethod
    def split_db_2to1(
            D: np.ndarray, L: np.ndarray, seed=0
    ) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
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
    def compute_mu(matrix: np.ndarray) -> np.ndarray:
        return MathHelper.v_col(matrix.mean(1))

    @staticmethod
    def center_matrix(matrix: np.ndarray) -> np.ndarray:
        mu = MathHelper.compute_mu(matrix)

        return matrix - mu

    @staticmethod
    def cv_matrix(matrix: np.ndarray) -> np.ndarray:
        centered_matrix: np.ndarray = MathHelper.center_matrix(matrix)
        # determine the main dimension of the matrix
        len_matrix = (
            float(matrix.shape[0])
            if float(matrix.shape[0]) > float(matrix.shape[1])
            else float(matrix.shape[1])
        )
        return np.dot(centered_matrix, centered_matrix.T) / len_matrix

    @staticmethod
    def compute_mu_and_sigma(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return MathHelper.compute_mu(matrix), MathHelper.cv_matrix(matrix)

    @staticmethod
    def inv_matrix(matrix: np.ndarray) -> np.ndarray:
        return np.linalg.inv(matrix)

    @staticmethod
    def det_matrix(matrix: np.ndarray) -> float:
        return np.linalg.det(matrix)

    @staticmethod
    def log_det_matrix(matrix: np.ndarray) -> float:
        return np.linalg.slogdet(matrix)[1]

    @staticmethod
    def mean(matrix: np.ndarray, axis: int = 1) -> np.ndarray:
        return np.mean(matrix, axis=axis)

    @staticmethod
    def var(matrix: np.ndarray, axis: int = 1) -> np.ndarray:
        return np.var(matrix, axis=axis)


class TimeHelper:
    """
    A helper class for measuring execution time.

    Attributes:
        start_time (float): The start time of the timer.
        end_time (float): The end time of the timer.
        print_time (bool): Flag indicating whether to print the execution time.
        delta_time (float): The difference between the start and end time.

    Methods:
        start_timer(): Starts the timer and returns the start time.
        end_timer(print_time=True): Ends the timer, calculates the execution time,
            and optionally prints it.

    """

    def __init__(self, print_time: bool = True) -> None:
        self.start_time: float = timeit.default_timer()
        self.end_time: float = None
        self.print_time: float = print_time
        self.delta_time: float = None

    def start_timer(self) -> float:
        """
        Starts the timer and returns the start time.

        Returns:
            float: The start time of the timer.

        """
        self.start_time: float = timeit.default_timer()
        return self.start_time

    def end_timer(self, print_time=True) -> float:
        """
        Ends the timer, calculates the execution time, and optionally prints it.

        Args:
            print_time (bool, optional): Flag indicating whether to print the execution time.
                Defaults to True.

        Returns:
            float: The execution time in seconds.

        """
        self.end_time: float = timeit.default_timer()
        self.delta_time: float = self.end_time - self.start_time
        if print_time:
            print(f"Execution time: {self.delta_time} seconds")
        return self.delta_time
