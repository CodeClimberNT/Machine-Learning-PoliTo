import os
from typing import Any, Optional
import numpy as np
import sklearn.datasets as datasets  # type: ignore

from .package_directory import config as cfg


class DatasetImporterHelper:
    @staticmethod
    def load_iris() -> tuple[Any, Any]:
        iris_dataset = datasets.load_iris()
        x, y = iris_dataset.data.T, iris_dataset.target

        return x, y

    @staticmethod
    def __get_train_data_path() -> str:
        return os.path.join(cfg.PROJECT_DIR_PATH, "trainData.txt")

    @staticmethod
    def load_train_project() -> tuple[np.ndarray, np.ndarray]:
        train_data_path: str = DatasetImporterHelper.__get_train_data_path()

        if DatasetImporterHelper.file_exists(train_data_path):
            x, y = DatasetImporterHelper.load_txt(train_data_path, features_type=float)
            return x, y
        else:
            raise FileNotFoundError(f"File not found: {train_data_path}")

    @staticmethod
    def load_train_project_splitted() -> (
        tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
    ):
        train_data_path: str = DatasetImporterHelper.__get_train_data_path()

        if DatasetImporterHelper.file_exists(train_data_path):
            x, y = DatasetImporterHelper.load_txt(train_data_path, features_type=float)
            from src.helpers import DataHandler as dh

            return dh.split_db_2to1(x, y)

        else:
            raise FileNotFoundError(f"File not found: {train_data_path}")

    @staticmethod
    def load_txt(
        file_path: str,
        sep: str = ",",
        labels_need_map: bool = False,
        labels_map: Optional[dict[str | int, int]] = None,
        features_type: type = float,
        labels_type: type = np.int32,
    ) -> tuple[np.ndarray, np.ndarray]:
        from src.helpers import MathHelper

        data_list: list = []
        labels_list: list = []

        if labels_map is None and labels_need_map:
            print("Using default labels map for Iris dataset")
            labels_map = {
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

                    if labels_need_map and labels_map is not None:
                        label = labels_map[label]

                    labels_list.append(label)

                except FileNotFoundError as e:
                    print(f"File not found: {e}")

        return np.hstack(data_list, dtype=features_type), np.array(
            labels_list, dtype=labels_type
        )

    @staticmethod
    def load_div_commedia(
        split_data: bool = False, n=0
    ) -> tuple[list[str], list[str], list[str]]:
        inferno_path: str = os.path.join(cfg.DIV_COMM_DIR_PATH, "inferno.txt")
        purgatorio_path: str = os.path.join(cfg.DIV_COMM_DIR_PATH, "purgatorio.txt")
        paradiso_path: str = os.path.join(cfg.DIV_COMM_DIR_PATH, "paradiso.txt")
        l_inf = []
        l_pur = []
        l_par = []

        paths = [inferno_path, purgatorio_path, paradiso_path]
        div_commedia = (l_inf, l_pur, l_par)

        for i, path in enumerate(paths):
            if DatasetImporterHelper.file_exists(path):
                with open(path, "r", encoding="ISO-8859-1") as f:
                    for lines in f:
                        div_commedia[i].append(lines)
            else:
                raise FileNotFoundError(f"File not found: {path}")

        return div_commedia

    # @staticmethod
    # def load_txt_with_np(
    #         file_path: str,
    #         delimiter: str = ",",
    #         features_type: type = float,
    #         labels_type: type = float,
    #         return_labels_in_data_matrix: bool = False,
    # ) -> tuple[np.ndarray, np.ndarray]:

    #     # Load the array from the text file and convert all but last column to float
    #     data = np.loadtxt(file_path, delimiter=delimiter, dtype=features_type)

    #     if return_labels_in_data_matrix:
    #         x = data
    #     else:
    #         # Extract features (all but last column)
    #         x = data[:, :-1]

    #     # Extract labels (last column)
    #     y = data[:, -1].astype(labels_type)

    #     return x, y

    @staticmethod
    def file_exists(file_path):
        return os.path.isfile(file_path)
