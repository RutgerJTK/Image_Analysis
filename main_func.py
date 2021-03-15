from src import split_stratified

def main():
    filename_labels = "data/Data_Entry_2017.csv"
    train_test_split(filename_labels)

def train_test_split(filename_labels):
    split_stratified.call_stratified(filename_labels)


main()