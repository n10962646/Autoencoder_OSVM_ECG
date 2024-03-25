# ASCREM
## Evaluating the Role of Multimodal Clinical Data in Breast Cancer Diagnostic Classifier

Author:Paul Yomer Ruiz Pinto (N10962646)
Last Modified: 25/03/2024<br/>
Code adapted from (ECGAD)[https://github.com/MediaBrain-SJTU/ECGAD/tree/main] which is licensed under MIT License permitting modification and distribution.

--------------

## Project Abstract

In the realm of electrocardiogram (ECG) anomaly detection, where accurate identification of irregularities is crucial for effective clinical intervention, this study aims to replicate and extend the findings of a sophisticated multi-scale cross-restoration framework proposed previously. This framework utilizes a deep autoencoder architecture to detect abnormal ECG signals by integrating both local and global characteristics, mirroring the diagnostic process of experienced cardiologists. However, despite its potential, the model faces challenges in interpretability and sensitivity to outliers, limiting its clinical applicability and performance reliability. Addressing these gaps, our investigation focuses on refining anomaly detection criteria and enhancing preprocessing techniques to mitigate the impact of outliers on model performance. Specifically, we aim to refine raw ECG signals by noise elimination, introduce single-channel short ECG signals using autoencoders, and evaluate the proposed model's performance using various metrics. By comparing it with the existing model, we seek to contribute to the advancement of anomaly detection methodologies in ECG analysis, ultimately improving diagnostic accuracy and patient care. The source codes for the experiments are available at https://github.com/MediaBrain-SJTU/ECGAD/tree/main (existing model) and https://github.com/n10962646/ASCREM.git (study model, initial stage).  
## Usage

### 1. Data Preparation <br/>
`data_preparation.ipynb` was used to prepare non-image features including feature selection and one-hot encoding, merge non-image and image data sources, perform stratified sampling and train test splits, and finally export data including ID references and meta feature names to the `\dataset` directory in the following file structure:
```bash
├── dataset
│   ├── train
│   │   ├── 1_x_cropped.npy
│   │   ├── 1_y.npy
│   │   ├── 1_meta.npy
│   │   ├── ...
│   ├── val
│   │   ├── 1_x_cropped.npy
│   │   ├── 1_y.npy
│   │   ├── 1_meta.npy
│   │   ├── ...
│   ├── test
│   │   ├── 1_x_cropped.npy
│   │   ├── 1_y.npy
│   │   ├── 1_meta.npy
│   │   ├── ...
│   ├── id_reference_test.txt
│   ├── id_reference_train.txt
│   ├── id_reference_val.txt
│   ├── meta_header.txt
```
where `*_x_cropped.npy` contains a raw cropped image of shape `(h, w)`, `*_y.npy` contains the associated target of shape `(1,)`, `*_meta.npy` contains the associated non-image data of shape `(n_features,)`, `id_reference_*.txt` contains the ID references per associated dataset and `meta_header.txt` contains the column names the final one-hot encoded non-image features.

### 2. Executing Pipelines <br/>

Assuming you have Anaconda installed, run `conda env create -f mri_fusion.yaml` to install all pre-requisites. The pipeline used for this work was simply to normalize and resize the image data (`pre_process.py`) and train models (`train.py`); you can find the specific commands to conduct all experiments in `run_experiments.bat` containing five experiments for each model at different number-generating seeds. After training, we conducted a feature importance analysis (`feature_imp.py`) to understand the contribution of each non-image features to the breast cancer prediction. Lastly, `plotROC.ipynb` was used to produce the combined ROC curves for the best run and five-run ensemble. All the experimental results for the two unimodal baselines and Learned Feature Fusion and Feature Fusion can be found in `results/` directory. 

To execute the experiments, simply run `train.py` with the arguments of your choice. Below is an example of training the Learned Feature Fusion model on the dataset with the settings used in the project using default values for most arguments (refer to `train.py` for more default values):
```python
python train.py --data_dir <path_to_your_dataset> \
                --out_dir <path_to_results> \
                --model learned-feature-fusion \
                --seed 0  
```

Example to run the learned feature fusion on the above trained model:
```python
python feature_imp.py   --model learned-feature-fusion
                        --model_name learned-feature-fusion-concat_no-CW_aug_seed0
```
