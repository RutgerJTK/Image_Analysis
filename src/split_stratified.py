from skmultilearn.model_selection import IterativeStratification, iterative_train_test_split
from sklearn.model_selection import train_test_split
import pandas
import pickle, os, shutil

def call_stratified(labels_df):
    iterative_train_test_split = iterative_train_test_split_new
    foto_names = labels_df["Image Index"]
    labels = labels_df["Finding_Labels"]
    # list_test = list(labels.head(1015))
    # print(list_test)
    # pickle.dump(list_test, open("data/test/labels.pk", "wb"))

    x_train, x_test_init, y_train, y_test_init = train_test_split(foto_names, labels,
                                                    test_size=0.2,
                                                    random_state=0)

    x_test, x_val, y_test, y_val = train_test_split(x_test_init, y_test_init,
                                                    test_size=0.5,
                                                    random_state=0)
    try:
        os.mkdir("data/train")
        os.mkdir("data/test")
        os.mkdir("data/val")
    except FileExistsError:
        pass


    for filename in x_train:
        try:
            shutil.move("data/images/" + filename, "data/train")
        except shutil.Error:
            # Als het bestand er al staat
            continue

    pickle.dump(list(y_train), open("data/train/labels.pk", "wb"))

    for filename in x_test:
        try:
            shutil.move("data/images/" + filename, "data/test")
        except shutil.Error:
            continue

    pickle.dump(list(y_train), open("data/test/labels.pk", "wb"))

    for filename in x_val:
        try:
            shutil.move("data/images/" + filename, "data/val")
        except shutil.Error:
            continue
    pickle.dump(list(y_train), open("data/val/labels.pk", "wb"))



def iterative_train_test_split_new(X, y, test_size):
    stratifier = IterativeStratification(n_splits=2, order=2, sample_distribution_per_fold=[test_size, 1.0-test_size])
    train_indexes, test_indexes = next(stratifier.split(X, y))

    X_train, y_train = X.iloc[train_indexes], y[train_indexes, :]
    X_test, y_test = X.iloc[test_indexes], y[test_indexes, :]

    return X_train, y_train, X_test, y_test