import argparse

from autoencoder import build_autoencoder, reconstruction_error_loss
from pre_process import load_data, preprocess_data
from graphs import plot_training_history, plot_original_and_reconstructed_global, plot_original_and_reconstructed_local, plot_anomaly_scores_histogram, plot_confusion_matrix_and_metrics


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.layers import Dense, Conv1D, BatchNormalization, Dropout, Flatten, Input, Concatenate, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout
from tensorflow.keras import layers, models, optimizers, losses
from tensorflow.keras.layers import BatchNormalization
import tensorflow as tf

import pandas as pd
import os
import scipy.io
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from biosppy.signals import ecg
from scipy import signal as ss

from keras.utils import model_to_dot, plot_model

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

from tqdm import tqdm
from collections import Counter

import warnings

warnings.filterwarnings('ignore')

tf.config.set_visible_devices([], 'GPU')


def train(global_layers=20, local_layers=4, epochs=30, batch_size=32, output_directory="./output", early_stopping_monitor='val_loss', early_stopping_patience=5, early_stopping_restore_best_weights=True):

    x_train, x_test, y_train, y_test = load_data()
    x_train_global, x_train_local = preprocess_data(x_train)
    x_test_global, x_test_local = preprocess_data(x_test)

    autoencoder = build_autoencoder(global_layers, local_layers)
    autoencoder.compile(optimizer='adam', loss={'output_global': reconstruction_error_loss, 'output_local': reconstruction_error_loss})
    
    dummy_latent_train = np.zeros((x_train_global.shape[0], 128))
    dummy_latent_test = np.zeros((x_test_global.shape[0], 128))

    early_stopping = EarlyStopping(monitor=early_stopping_monitor, patience=early_stopping_patience, restore_best_weights=early_stopping_restore_best_weights)
    
    history = autoencoder.fit(
        [x_train_global, x_train_local],  
        [x_train_global, x_train_local, dummy_latent_train],  # Targets
        epochs=epochs,
        batch_size=batch_size, 
        validation_data=(
            [x_test_global, x_test_local],  
            [x_test_global, x_test_local, dummy_latent_test]   # Validation targets 
        ), 
        callbacks=[early_stopping]
    )

    plot_training_history(history, output_directory)

    plot_original_and_reconstructed_global(autoencoder, x_test_global, x_test_local, y_test, output_directory)
    plot_original_and_reconstructed_local(autoencoder, x_test_global, x_test_local, y_test, output_directory)

    encoded_model = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('latent_space').output)
    encoded_train = encoded_model.predict([x_train_global, x_train_local])
    encoded_test = encoded_model.predict([x_test_global, x_test_local])

    osvm = OneClassSVM(kernel='rbf', nu=0.1)  
    osvm.fit(encoded_train)

    osvm_scores = osvm.decision_function(encoded_test)

    plot_anomaly_scores_histogram(osvm_scores, output_directory)
    plot_confusion_matrix_and_metrics(osvm_scores, y_test, output_directory)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the autoencoder model.")
    parser.add_argument("--global_layers", type=int, default=20, help="Number of layers in the global branch.")
    parser.add_argument("--local_layers", type=int, default=4, help="Number of layers in the local branch.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs for training.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--output_directory", type=str, default="./output", help="Directory to save the training results.")
    parser.add_argument("--early_stopping_monitor", type=str, default="val_loss", help="Quantity to be monitored for early stopping.")
    parser.add_argument("--early_stopping_patience", type=int, default=5, help="Number of epochs with no improvement after which training will be stopped.")
    parser.add_argument("--early_stopping_restore_best_weights", type=bool, default=True, help="Whether to restore model weights from the epoch with the best value of the monitored quantity.")

    args = parser.parse_args()

    train(args.global_layers, args.local_layers, args.epochs, args.batch_size, args.output_directory, args.early_stopping_monitor, args.early_stopping_patience, args.early_stopping_restore_best_weights)
