import os
import pickle
import time

import cv2
import numpy
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.models import Model, Sequential, load_model, model_from_json
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from keras.metrics import Precision, Recall
from sklearn.preprocessing import MultiLabelBinarizer
from keras import backend as K


K.tensorflow_backend._get_available_gpus()
#check if gpu word gebruikt

num_classes = 15
unique_labels = ['Cardiomegaly', 'Emphysema', 'Effusion', 'No Finding', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Atelectasis', 'Pneumothorax', 'Pleural_Thickening', 'Pneumonia', 'Fibrosis', 'Edema', 'Consolidation']
def main():
    train_model()
    load_model()

def dataset_load(dirname):
    print("Started loading the dataset... ({})".format(dirname))

    start = time.time()

    # Labels first
    labels = pickle.load(open("{}/labels.pk".format(dirname), "rb"))
    mlb = MultiLabelBinarizer()
    mlb = mlb.fit([unique_labels])
    labels = [mlb.transform([label]) for label in labels]
    labels = numpy.array(labels)
    print(mlb.classes_)
    print(labels[1])

    # Images next
    image_path = dirname
    image_filenames = sorted([os.path.join(image_path, file)
        for file in os.listdir(image_path) if file.endswith('.png')])
    images = [preprocess(i) for i in image_filenames]

    images = numpy.array(images)

    print("Dataset is loaded, took: ", time.time()-start)
    return images, labels

def preprocess(image_path):
    img = image.load_img(image_path, target_size=(240, 240))
    img = image.img_to_array(img)
    img = numpy.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def train_model():
    print("Loading the model...")
    base_model = ResNet50(weights= None,
                    # Zodat we niet vastzitten aan de eigenschappen van de inputs van pretrainen
                    include_top=False,
                    # De vaststaande grootte en 3
                    input_shape= (240, 240, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.7)(x)
    predictions = Dense(num_classes, activation= 'sigmoid')(x)
    model = Model(inputs = base_model.input, outputs = predictions)

    
    # sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    adam = Adam(lr=0.0001)
    model.compile(optimizer= adam, loss='categorical_crossentropy', metrics=['accuracy'])

    photos, labels = dataset_load("data/train")
    X_train = photos.reshape(-1, 240, 240, 3)
    Y_train = labels.reshape(-1, 15)
    photos, labels = dataset_load("data/val")
    X_val = photos.reshape(-1, 240, 240, 3)
    Y_val = labels.reshape(-1, 15)

    es = EarlyStopping(monitor='val_loss', mode='min', patience=5, baseline=0.6)
    model.fit(X_train, Y_train, epochs = 2000, batch_size = 64, validation_data=(X_val, Y_val), callbacks=[es])

    # serialize model to JSON
    model_json = model.to_json()
    with open("model/model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model/model.h5")
    print("Saved model to disk")

def load_model():
    # load json and create model
    json_file = open('model/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model/model.h5")
    print("Loaded model from disk")
    
    # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy', Precision(), Recall()])
    
    photos, labels = dataset_load("data/test")
    X_test = photos.reshape(-1, 240, 240, 3)
    Y_test = labels.reshape(-1, 15)
    score = loaded_model.evaluate(X_test, Y_test, verbose=0)

    print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
    print(loaded_model.metrics_names, score)

    print(loaded_model.predict(X[1]))

main()
