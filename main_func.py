from src import split_stratified, label_checker, organize_images
import pandas
import collections
import matplotlib.pyplot as plt
import numpy as np

def main():
    # filename_labels = "D:\School - all things school related\HAN Bio-informatica\Jaar 3\Bi10\Image Analysis\VS_ImageAnalysis\Image_Analysis\data\Data_Entry_2017.csv"
    filename_labels = "C:\\Users\\Wouter\\Documents\\GitHub\\Image_Analysis\\data\\Data_Entry_2017.csv"
    # try:
    #     filename_labels = "/data/Data_Entry_2017.csv"   	   ## pathing problem
    # except FileNotFoundError:
    #     filename_labels = "D:\School - all things school related\HAN Bio-informatica\Jaar 3\Bi10\Image Analysis\VS_ImageAnalysis\Image_Analysis\data\Data_Entry_2017.csv" ## fixt rutger's path problem
    labels_file = pandas.read_csv(filename_labels, sep=',', engine="python")
    labels_file["Finding_Labels"] = labels_file["Finding_Labels"].map(lambda Finding_Labels: Finding_Labels.split("|"))     ## bevat de labels benodigd voor classificatie ~ röttie
    
    #organize_images.tar_to_jpeg()
    train_test_split(labels_file)
    check_labels(labels_file)

def train_test_split(labels_file):
    x_train, y_train, x_test, y_test, x_val, y_val = split_stratified.call_stratified(labels_file)
    organize_images.images_to_folders(x_train, y_train, x_test, y_test, x_val, y_val)


def check_labels(labels_file):
    label_checker.label_go_check(labels_file)
    print('I did sumthin and it sorta wurks ¯\_(ツ)_/¯ yay I guess!')

main()