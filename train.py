import numpy as np
import mat73
import sklearn.metrics
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Reshape
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Conv3D, MaxPooling3D, BatchNormalization, Add, ZeroPadding2D
import matplotlib.pyplot as plt
from preprocess import process_data
from model import create_vit_classifier, run_experiment, learning_rate

if "__name__"=="__main__":
    train_data, train_labels_reshaped, test_data, test_labels_reshaped, final_test, final_labels_reshaped = process_data()

    # Store the unseen test data
    np.save("saved_model/test.npy", final_test)
    np.save("save_model/test_label.npy", final_labels_reshaped)
    
    # To run the model on a list of learning rates
    for lr in learning_rate:
        vit_classifier = create_vit_classifier()

        optimizer = keras.optimizers.Adam(
            learning_rate=lr#, weight_decay=weight_decay
        )
        vit_classifier.compile(
            optimizer=optimizer,
            loss=keras.losses.CategoricalCrossentropy(),
            metrics=[
                keras.metrics.CategoricalAccuracy(name="accuracy"),
            ],
        )
        history_exp = run_experiment(vit_classifier, train_data, train_labels_reshaped, test_data, test_labels_reshaped)

    # SAVE MODEL
    vit_classifier.save("saved_model/best_vit_model")
    vit_classifier.save_weights("saved_model/vit_best.h5")

    