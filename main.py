import librosa
import matplotlib.pyplot as plt
import librosa.display
import os
import glob
import cv2

def process_data():
    spectrogram_file = "spectrogram"
    wavelets_file = "wavelets"
    spectrogram_train = []
    wavelets_train = []
    spectrogram_test = []
    wavelets_test = []

    spectrogram_train_file = os.path.join(spectrogram_file, "train")
    class_idx = 0
    for class_name in glob.glob(os.path.join(spectrogram_train_file, "*")):
        for image in glob.glob(os.path.join(class_name, "*")):
            spectrogram_train.append((cv2.imread(image), class_idx)) # fix image read
        class_idx+=1
    spectrogram_test_file = os.path.join(spectrogram_file, "test")
    class_idx = 0
    for class_name in glob.glob(os.path.join(spectrogram_test_file, "*")):
        for image in glob.glob(os.path.join(class_name, "*")):
            spectrogram_test.append((cv2.imread(image), class_idx)) # fix image read
        class_idx+=1

    wavelets_train_file = os.path.join(wavelets_file, "train")
    class_idx = 0
    for class_name in glob.glob(os.path.join(wavelets_train_file, "*")):
        for image in glob.glob(os.path.join(class_name, "*")):
            wavelets_train.append((cv2.imread(image), class_idx)) # fix image read
        class_idx+=1

    wavelets_test_file = os.path.join(wavelets_file, "test")
    class_idx = 0
    for class_name in glob.glob(os.path.join(wavelets_test_file, "*")):
        for image in glob.glob(os.path.join(class_name, "*")):
            wavelets_test.append((cv2.imread(image), class_idx)) # fix image read
        class_idx+=1

    train = list(zip(wavelets_train, spectrogram_train))
    test = list(zip(wavelets_test, spectrogram_test))
    # train is a list of tuples, each tuple has two baby tuples, each baby tuple has (image (wavelets/spectrogram), class_idx)


    def train_model():
        train, test = process_data()

process_data()