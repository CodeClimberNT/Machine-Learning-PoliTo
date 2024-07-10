import os
from pathlib import Path
import numpy as np


class DatasetImporterHelper:
    @staticmethod
    def load_iris():
        import sklearn.datasets as datasets
        iris_dataset = datasets.load_iris()
        x, y = iris_dataset["data"].T, iris_dataset["target"]

        return x, y

    @staticmethod
    def load_train_project():
        current_file_path: str = os.path.abspath(__file__)
        project_root: str = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
        train_data_path: str = os.path.join(project_root, 'datasets', 'project', 'trainData.txt')

        if DatasetImporterHelper.file_exists(train_data_path):
            return DatasetImporterHelper.load_txt(
                train_data_path, features_type=float
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
        from src.helpers import MathHelper
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
