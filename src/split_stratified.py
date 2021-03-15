import pandas
from skmultilearn.model_selection import IterativeStratification

def call_stratified(labels_filename):
    labels_file = pandas.read_csv(labels_filename, sep=',', engine="python")
    print(list(labels_file.columns))
    k_fold = IterativeStratification(n_splits=2, order=1)
    # for train, test in k_fold.split(labels_file["Image Index"], labels_file["Finding Labels"]):
    #     print(train, test)
