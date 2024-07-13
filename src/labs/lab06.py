from src.helpers import DatasetImporterHelper as dih, DataHandler as dh


def sol1_multi_class_classification():
    l_inf, l_purg, l_par = dih.load_div_commedia()
    l_inf_train, l_inf_val = dh.split_data(l_inf, 4)
    l_purg_train, l_purg_val = dh.split_data(l_inf, 4)
    l_par_train, l_par_val = dh.split_data(l_inf, 4)
    print(len(l_inf_train))


def main():
    sol1_multi_class_classification()


if __name__ == "__main__":
    main()
