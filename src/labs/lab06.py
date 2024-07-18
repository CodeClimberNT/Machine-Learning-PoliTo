from typing import Any
from src.helpers import DatasetImporterHelper as dih, DataHandler as dh
from src.models.gaussian_models.multinomial_model import MultinomialModel as MM
import numpy as np


def sol1_multi_class_classification():
    l_inf, l_purg, l_par = dih.load_div_commedia()
    l_inf_train, l_inf_val = dh.split_data(l_inf, 4)
    l_purg_train, l_purg_val = dh.split_data(l_inf, 4)
    l_par_train, l_par_val = dh.split_data(l_inf, 4)

    cls_to_idx: dict[str, int] = {"inferno": 0, "purgatorio": 1, "paradiso": 2}

    tercets_train_dict: dict[str, list[Any]] = {
        "inferno": l_inf_train,
        "purgatorio": l_purg_train,
        "paradiso": l_par_train,
    }

    tercets_eval_list = np.array(l_inf_val + l_purg_val + l_par_val)

    mm_model = MM().fit(tercets_train_dict)

    log_posteriors = mm_model.compute_class_posteriors(tercets_eval_list)
# need to fix


def main():
    sol1_multi_class_classification()


if __name__ == "__main__":
    main()
