import os
import glob
import numpy as np

from src.helpers import (
    DatasetImporterHelper as dih,
    DataHandler as dh,
    ModelEvaluator as me,
)
from src.models.gaussian_models import (
    MultivariateGaussianModel as MVG,
    NaiveBayesBaseGaussianModel as NBG,
    
)

from src.models.gaussian_models.gaussian_utils import GaussianUtils as gauss_utils

current_file_path: str = os.path.abspath(__file__)
project_root: str = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
commedia_ll_path = glob.glob("**/commedia_ll.npy", recursive=True)[0]
commedia_labels_path = glob.glob("**/commedia_labels.npy", recursive=True)[0]
commedia_llr_infpar_path = glob.glob("**/commedia_llr_infpar.npy", recursive=True)[0]
commedia_labels_infpar_path = glob.glob(
    "**/commedia_labels_infpar.npy", recursive=True
)[0]
commedia_llr_infpar_eps1_path = glob.glob(
    "**/commedia_llr_infpar_eps1.npy", recursive=True
)[0]
commedia_labels_infpar_eps1_path = glob.glob(
    "**/commedia_labels_infpar_eps1.npy", recursive=True
)[0]
commedia_ll_eps1_path = glob.glob("**/commedia_ll_eps1.npy", recursive=True)[0]
commedia_labels_eps1_path = glob.glob("**/commedia_labels_eps1.npy", recursive=True)[0]

commedia_ll = np.load(os.path.join(project_root, commedia_ll_path))
commedia_labels = np.load(os.path.join(project_root, commedia_labels_path))
commedia_llr_infpar = np.load(os.path.join(project_root, commedia_llr_infpar_path))
commedia_labels_infpar = np.load(
    os.path.join(project_root, commedia_labels_infpar_path)
)
commedia_llr_infpar_eps1 = np.load(
    os.path.join(project_root, commedia_llr_infpar_eps1_path)
)
commedia_labels_infpar_eps1 = np.load(
    os.path.join(project_root, commedia_labels_infpar_eps1_path)
)
commedia_ll_eps1 = np.load(os.path.join(project_root, commedia_ll_eps1_path))
commedia_labels_eps1 = np.load(os.path.join(project_root, commedia_labels_eps1_path))


def confusion_matrix_mvg():
    print()
    print("Multiclass - uniform priors and costs - confusion matrix")

    commedia_posteriors = gauss_utils.compute_posteriors_from_prob(commedia_ll, np.ones(3)/3.0)
    commedia_predictions = me.compute_optimal_Bayes(commedia_posteriors, me.uniform_cost_matrix(3))

    print(me.compute_confusion_matrix(commedia_predictions, commedia_labels))


def main():
    confusion_matrix_mvg()


if __name__ == "__main__":
    main()
