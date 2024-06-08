import os
import numpy as np
from tensorflow.keras.layers import Input, Dense, Conv1D, Flatten, BatchNormalization, Dropout, Reshape, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
from tqdm import tqdm
import argparse

def global_branch(input_global, num_layers):
    encoded_global = input_global
    for _ in range(num_layers):
        encoded_global = Dense(1024, activation='relu')(encoded_global)
        encoded_global = BatchNormalization()(encoded_global)  
        encoded_global = Dropout(0.5)(encoded_global)  

    encoded_global = Dense(256, activation='relu')(encoded_global)
    encoded_global = BatchNormalization()(encoded_global)

    decoded_global = Dense(256, activation='relu')(encoded_global)
    decoded_global = Dense(512, activation='relu')(decoded_global)
    decoded_global = Dense(9000, activation='sigmoid', name='output_global')(decoded_global)
    
    return encoded_global, decoded_global

def local_branch(input_local, num_layers):
    encoded_local = input_local
    for _ in range(num_layers):
        encoded_local = Conv1D(64, 3, activation='relu')(encoded_local)
        encoded_local = BatchNormalization()(encoded_local)  
        encoded_local = Conv1D(128, 3, activation='relu')(encoded_local)
        encoded_local = BatchNormalization()(encoded_local)  

    encoded_local = Flatten()(encoded_local)
    encoded_local = Dense(512, activation='relu')(encoded_local)
    encoded_local = Dropout(0.5)(encoded_local)  

    decoded_local = Dense(512, activation='relu')(encoded_local)
    decoded_local = Dense(179*100, activation='sigmoid')(encoded_local)
    decoded_local = Reshape((179, 100))(decoded_local)
    decoded_local = Conv1D(128, 3, activation='relu', padding='same')(decoded_local)
    decoded_local = Conv1D(64, 3, activation='relu', padding='same')(decoded_local)
    decoded_local = Conv1D(1, 3, activation=None, padding='same', name='output_local')(decoded_local)
    return encoded_local, decoded_local

def build_autoencoder(global_layers, local_layers):
    input_global = Input(shape=(9000,), name='input_global')
    input_local = Input(shape=(179, 100), name='input_local')

    encoded_global, decoded_global = global_branch(input_global, global_layers)
    encoded_local, decoded_local = local_branch(input_local, local_layers)

    merged = Concatenate()([encoded_global, encoded_local])

    latent = Dense(128, activation='relu', name='latent_space')(merged)
    
    autoencoder = Model(inputs=[input_global, input_local],
                        outputs=[decoded_global, decoded_local, latent])
    return autoencoder

def reconstruction_error_loss(y_true, y_pred):
    return K.mean(K.abs(y_true - y_pred))  # Mean Absolute Error (MAE)

def save_model_plot(model, output_directory):
    os.makedirs(output_directory, exist_ok=True)
    plot_path = os.path.join(output_directory, "model_plot.png")
    plot_model(model, to_file=plot_path, show_shapes=True, dpi=60)
    print(f"Model plot saved to {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build an autoencoder model and save its plot.")
    parser.add_argument("output_directory", type=str, help="Directory to save the model plot image.")
    parser.add_argument("--global_layers", type=int, default=20, help="Number of layers in the global branch.")
    parser.add_argument("--local_layers", type=int, default=4, help="Number of layers in the local branch.")

    args = parser.parse_args()