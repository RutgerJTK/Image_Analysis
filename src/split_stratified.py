import pandas
from skmultilearn.model_selection import iterative_train_test_split

def call_stratified(labels_filename):
    labels_file = pandas.read_csv(labels_filename, sep=',', engine="python")
    labels_file["Finding_Labels"] = labels_file["Finding_Labels"].map(lambda Finding_Labels: Finding_Labels.split("|"))
    print("kip")
