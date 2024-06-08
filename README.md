# MCRF-SVM
## Enhancing ECG Anomaly Detection: A Multi-scale Cross-restoration Framework Integrated with One-class SVM (MCRF-SVM)

Author:Paul Yomer Ruiz Pinto (N10962646)
Last Modified: 08/06/2024<br/>
Code adapted from [ECGAD](https://github.com/MediaBrain-SJTU/ECGAD/tree/main) which is licensed under MIT License permitting modification and distribution.

--------------

## Project Abstract

In the realm of medical diagnostics, the electrocardiogram (ECG) is an essential tool for detecting heart diseases by capturing the heart's electrical activity. ECG signals, a type of time series data, are crucial for identifying abnormalities such as arrhythmias, heart attacks, and heart failure. Despite advancements in automated ECG analysis, the scarcity of labelled abnormal samples poses a significant challenge for supervised anomaly detection models. This scarcity affects the accuracy and effectiveness of these models, highlighting the need for unsupervised techniques that can efficiently handle limited labelled data. This research introduced a novel multi-scale cross-restoration framework combined with one-class Support Vector Machines (SVM) to enhance the discriminative power of hybrid anomaly detection models for ECG signals. The framework utilized a two-branch autoencoder architecture: one branch captured global features from the entire ECG, while the other focused on local, heartbeat-level details. The approach was evaluated using an adapted version of the PhysioNet/CinC Challenge 2017 database. The proposed model achieved an accuracy of 64.0%, a precision of 77.0%, a recall of 75.0%, and an F1-score of 76.0%. These results demonstrated balanced and robust performance in classifying ECG anomalies. Compared to existing methods, such as deep neural networks and hybridized approaches, our model's integration of multi-scale restoration with SVMs enhanced robustness against diverse anomalies. While the accuracy was lower than some benchmarks, the approach showed competitive precision and recall rates. The integration of a multi-scale cross-restoration framework with one-class SVMs showed potential in enhancing the discriminative power of hybrid anomaly detection models for ECG signals. Although further optimization is necessary to achieve higher accuracy, the current results underscore the potential utility of this approach in clinical diagnosis. This study lays the groundwork for subsequent research to build upon these findings, aiming for improved detection accuracy and broader implementation in clinical environments. 

## 1. Data Exploration <br/>
The notebook `EDA.ipynb` is utilized to determine the optimal preprocessing techniques, encompassing feature selection, and to convert the data into `.npy` files. The directory `/data` is currently kept private and organised as follows:
```bash
├── data
│   ├── training
│   │   ├── *.mat
│   │   ├── *.hea
│   │   ├── ...
│   ├── testing
│   │   ├── *.mat
│   │   ├── *.hea
│   │   ├── ...
│   ├── training.csv
│   ├── testing.csv
```
where, the ECG recordings were sampled at 300 Hertz and all data is provided in MATLAB V4 WFDB-compliant format, each including a `.mat` file containing the ECG and a `.hea` file containing the waveform information. 

## 2. Data Preparation

The following scripts are designed to process and prepare ECG signal data for machine learning tasks. The first script `mat_to_npy.py` converts `.mat` files into `.npy` format, ensuring that the signals are of a consistent length. The second script `npy_to_npz.py` further processes these `.npy`files by associating each signal with a label and saving the result in `.npz` format. The final script `load_npz.py` loads these `.npz` files, ensuring that the signals are padded or truncated to a specified length, and prepares them for further analysis or model training. 

Convert .mat Files to .npy Files:
```python
python mat_to_npy.py /path/to/mat/files /path/to/output/npy/files --signal_length 9000
```
Convert .npy Files to .npz Files:
```python
python npy_to_npz.py /path/to/npy/files /path/to/output/npz/files /path/to/labels.csv
```
Load and Process .npz Files:
```python
python load_npz.py /path/to/npz/files 9000
```

## 3. Model: ECG Signal Analysis with Autoencoder and One-class SVM

The following pipeline consists of several modular Python scripts designed to handle different stages of the analysis process, from data preprocessing to model training and evaluation.

**Preprocessing ECG Data**

The `pre_process.py` script serves as the initial step in the pipeline, facilitating the loading and preprocessing of ECG data stored in `.npy` files. Users can specify parameters such as window size and step size for segmenting the signals, allowing for fine-tuning of the preprocessing stage.

```python
python pre_process.py <input_directory> <output_directory> --window_size <window_size> --step_size <step_size>
```

**Autoencoder Model**

Once the data is preprocessed, the `autoencoder.py` script defines and compiles an autoencoder model architecture tailored for ECG signal analysis. This script offers flexibility in configuring the number of layers in both the global and local branches of the autoencoder, enabling users to customize the model architecture based on their specific requirements.

```python
python autoencoder.py <output_directory> --global_layers <global_layers> --local_layers <local_layers>
```
**Training Autoencoder and One-class SVM**

The `train.py` script orchestrates the training process of the autoencoder model using the preprocessed ECG data to pass the encoded data to the one-class SVM. Training parameters such as the number of epochs, batch size, and early stopping criteria to optimize model performance can be adjusted. 

```python
python train.py --global_layers <global_layers> --local_layers <local_layers> --epochs <epochs> --batch_size <batch_size> --output_directory <output_directory> --early_stopping_monitor <early_stopping_monitor> --early_stopping_patience <early_stopping_patience> --early_stopping_restore_best_weights <early_stopping_restore_best_weights>
```
**Evaluation of Anomaly Detection**

Upon completion of training, the trained autoencoder model is evaluated for its performance in reconstructing ECG signals and detecting anomalies. The pipeline generates comprehensive visualizations, including reconstructed signal plots, anomaly score histograms, and confusion matrices, enabling users to assess the model's efficacy in identifying abnormal cardiac activities accurately.

```python
python evaluate.py --output_directory <output_directory> --global_layers <global_layers> --local_layers <local_layers> --threshold <threshold>
```

## 4. References
```bash
@inproceedings{jiang2022ecgad,
  title={Multi-scale Cross-restoration Framework for Electrocardiogram Anomaly Detection}
  author={Jiang, Aofan and Huang, Chaoqin and Cao, Qing and Wu, Shuang and Zeng, Zi and Chen, Kang and Zhang, Ya and Wang, Yanfeng},
  booktitle={International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI)},
  year={2023}
}
```

