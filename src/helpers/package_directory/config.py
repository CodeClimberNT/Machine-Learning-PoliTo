import os

ROOT_DIR_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


DATASET_DIR_PATH = os.path.join(os.path.dirname(ROOT_DIR_PATH), "datasets")

DIV_COMM_DIR_PATH = os.path.join(DATASET_DIR_PATH, "generative", "div_comm")


PROJECT_DIR_PATH = os.path.join(DATASET_DIR_PATH, "project")
