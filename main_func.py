from src import split_stratified
import pandas

def main():
    filename_labels = "data/Data_Entry_2017.csv"
    labels_file = pandas.read_csv(filename_labels, sep=',', engine="python")
    labels_file["Finding_Labels"] = labels_file["Finding_Labels"].map(lambda Finding_Labels: Finding_Labels.split("|"))
 
    train_test_split(filename_labels)

def train_test_split(filename_labels):
    split_stratified.call_stratified(filename_labels)


main()