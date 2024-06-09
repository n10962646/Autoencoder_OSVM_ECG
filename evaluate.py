import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from tqdm import tqdm
import argparse
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns

def plot_training_history(history, output_directory):
    os.makedirs(output_directory, exist_ok=True)
    loss_plot_path = os.path.join(output_directory, "training_loss_plot.png")

    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"Training history plot saved to {loss_plot_path}")

def plot_original_and_reconstructed_global(autoencoder, x_test_global, x_test_local, y_test, output_directory):
    os.makedirs(output_directory, exist_ok=True)
    plot_path = os.path.join(output_directory, "original_and_reconstructed_global.png")

    def min_max_scaling(data):
        min_val = np.min(data)
        max_val = np.max(data)
        scaled_data = (data - min_val) / (max_val - min_val)
        return scaled_data

    normal_indices = np.where(y_test == 0)[0]
    abnormal_indices = np.where(y_test == 1)[0]
    selected_indices = np.random.choice(np.concatenate((normal_indices, abnormal_indices)), size=2, replace=False)

    reconstructed_global, _, _ = autoencoder.predict([x_test_global[selected_indices], x_test_local[selected_indices]])

    plt.figure(figsize=(20, 10))
    for i, index in enumerate(selected_indices):
        ax = plt.subplot(2, 2, i + 1)
        original_data = min_max_scaling(x_test_global[index])
        plt.plot(original_data, color='blue')
        label = 'Normal' if y_test[index] == 0 else 'Abnormal'
        plt.title(f"Original Global ({label})", fontsize=14)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.grid(True)

        ax = plt.subplot(2, 2, i + 3)
        reconstructed_data = min_max_scaling(reconstructed_global[i])
        plt.plot(reconstructed_data, color='green')
        plt.title(f"Reconstructed Global ({label})", fontsize=14)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.grid(True)

    plt.subplots_adjust(hspace=0.4)
    plt.savefig(plot_path)
    plt.close()
    print(f"Global reconstruction plot saved to {plot_path}")

def plot_original_and_reconstructed_local(autoencoder, x_test_global, x_test_local, y_test, output_directory):
    os.makedirs(output_directory, exist_ok=True)
    plot_path = os.path.join(output_directory, "original_and_reconstructed_local.png")

    def min_max_scaling(data):
        min_val = np.min(data)
        max_val = np.max(data)
        scaled_data = (data - min_val) / (max_val - min_val)
        return scaled_data

    normal_indices = np.where(y_test == 0)[0]
    abnormal_indices = np.where(y_test == 1)[0]
    selected_indices = np.random.choice(np.concatenate((normal_indices, abnormal_indices)), size=2, replace=False)

    _, reconstructed_local, _ = autoencoder.predict([x_test_global[selected_indices], x_test_local[selected_indices]])

    plt.figure(figsize=(20, 5))
    for i, index in enumerate(selected_indices):
        ax = plt.subplot(2, 2, i + 1)
        for j in range(x_test_local.shape[2]):
            plt.plot(x_test_local[index, :, j])
        label = 'Normal' if y_test[index] == 0 else 'Abnormal'
        plt.title(f"Original Local ({label})", fontsize=14)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.grid(True)
        if i == 0:
            plt.legend(loc='upper right')

        ax = plt.subplot(2, 2, i + 3)
        for j in range(reconstructed_local.shape[2]):
            plt.plot(min_max_scaling(reconstructed_local[i, :, j]))
        plt.title(f"Reconstructed Local ({label})", fontsize=14)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.grid(True)
        if i == 0:
            plt.legend(loc='upper right')

    plt.subplots_adjust(hspace=0.4)
    plt.savefig(plot_path)
    plt.close()
    print(f"Local reconstruction plot saved to {plot_path}")
    
    

def plot_anomaly_scores_histogram(osvm_scores, output_directory):
    os.makedirs(output_directory, exist_ok=True)
    plot_path = os.path.join(output_directory, "anomaly_scores_histogram.png")

    osvm_scores = osvm_scores.reshape(-1, 1)

    scaler = MinMaxScaler()
    osvm_scores_scaled = scaler.fit_transform(osvm_scores).flatten()

    percentiles = [1, 95]
    percentile_values = np.percentile(osvm_scores_scaled, percentiles)

    plt.figure(figsize=(15, 5))
    plt.hist(osvm_scores_scaled, bins=100, density=True, alpha=0.9, color='g')

    for p, value in zip(percentiles, percentile_values):
        plt.axvline(x=value, color='r', linestyle='--')
        plt.text(value, plt.ylim()[1]*0.8, f'{p}th', color='r', rotation=90)

    plt.xlabel('Anomaly Score')
    plt.ylabel('Density')
    plt.title('Distribution of Anomaly Scores')
    plt.savefig(plot_path)
    plt.close()
    print(f"Anomaly scores histogram saved to {plot_path}")

def plot_confusion_matrix_and_metrics(osvm_scores, y_test, output_directory, threshold=75):
    os.makedirs(output_directory, exist_ok=True)
    plot_path = os.path.join(output_directory, "confusion_matrix_and_metrics.png")

    threshold = np.percentile(osvm_scores, q=threshold)
    print("Threshold:", threshold)
    osvm_predicted = np.where(osvm_scores <= threshold, 1, 0)

    accuracy = accuracy_score(y_test, osvm_predicted)
    conf_matrix = confusion_matrix(y_test, osvm_predicted)

    f1 = f1_score(y_test, osvm_predicted, pos_label=1)

    print("Accuracy:", accuracy)
    print("F1 Score:", f1)

    precision = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])
    recall = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])

    print("Precision:", precision)
    print("Recall:", recall)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,               
                xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])

    plt.text(1, -0.2, f'Test F1 Score: {f1:.2f}', horizontalalignment='center', fontsize=16, color='Black')

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix for ECG')
    plt.savefig(plot_path)
    plt.close()
    print(f"Confusion matrix and metrics plot saved to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training history and reconstruction results.")
    parser.add_argument("output_directory", type=str, help="Directory to save the plots.")
    parser.add_argument("--global_layers", type=int, default=20, help="Number of layers in the global branch.")
    parser.add_argument("--local_layers", type=int, default=4, help="Number of layers in the local branch.")
    parser.add_argument("--threshold", type=int, default=75, help="Threshold for classifying anomalies (percentile)")

    args = parser.parse_args()
