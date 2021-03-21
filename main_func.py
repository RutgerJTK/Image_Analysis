from src import split_stratified, label_checker
import pandas

def main():
    try:
        filename_labels = "/data/Data_Entry_2017.csv"   	   ## pathing problem
    except FileNotFoundError:
        filename_labels = "D:\School - all things school related\HAN Bio-informatica\Jaar 3\Bi10\Image Analysis\VS_ImageAnalysis\Image_Analysis\data\Data_Entry_2017.csv" ## fixt rutger's path problem
    labels_file = pandas.read_csv(filename_labels, sep=',', engine="python")
    labels_file["Finding_Labels"] = labels_file["Finding_Labels"].map(lambda Finding_Labels: Finding_Labels.split("|"))     ## bevat de labels benodigd voor classificatie ~ röttie
 
    train_test_split(filename_labels)
    check_labels(labels_file)

def train_test_split(filename_labels):
    split_stratified.call_stratified(filename_labels)


def check_labels(labels_file):
    label_checker.label_go_check(labels_file)


main()