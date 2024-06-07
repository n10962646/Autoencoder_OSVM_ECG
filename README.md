# MCRF-SVM
## Enhancing ECG Anomaly Detection: A Multi-scale Cross-restoration Framework Integrated with One-class SVM (MCRF-SVM)

Author:Paul Yomer Ruiz Pinto (N10962646)
Last Modified: 08/06/2024<br/>
Code adapted from [ECGAD](https://github.com/MediaBrain-SJTU/ECGAD/tree/main) which is licensed under MIT License permitting modification and distribution.

--------------

## Project Abstract

In the realm of medical diagnostics, the electrocardiogram (ECG) is an essential tool for detecting heart diseases by capturing the heart's electrical activity. ECG signals, a type of time series data, are crucial for identifying abnormalities such as arrhythmias, heart attacks, and heart failure. Despite advancements in automated ECG analysis, the scarcity of labelled abnormal samples poses a significant challenge for supervised anomaly detection models. This scarcity affects the accuracy and effectiveness of these models, highlighting the need for unsupervised techniques that can efficiently handle limited labelled data. This research introduced a novel multi-scale cross-restoration framework combined with one-class Support Vector Machines (SVM) to enhance the discriminative power of hybrid anomaly detection models for ECG signals. The framework utilized a two-branch autoencoder architecture: one branch captured global features from the entire ECG, while the other focused on local, heartbeat-level details. The approach was evaluated using an adapted version of the PhysioNet/CinC Challenge 2017 database. The proposed model achieved an accuracy of 64.0%, a precision of 77.0%, a recall of 75.0%, and an F1-score of 76.0%. These results demonstrated balanced and robust performance in classifying ECG anomalies. Compared to existing methods, such as deep neural networks and hybridized approaches, our model's integration of multi-scale restoration with SVMs enhanced robustness against diverse anomalies. While the accuracy was lower than some benchmarks, the approach showed competitive precision and recall rates. The integration of a multi-scale cross-restoration framework with one-class SVMs showed potential in enhancing the discriminative power of hybrid anomaly detection models for ECG signals. Although further optimization is necessary to achieve higher accuracy, the current results underscore the potential utility of this approach in clinical diagnosis. This study lays the groundwork for subsequent research to build upon these findings, aiming for improved detection accuracy and broader implementation in clinical environments. 

## Inital Stage

### 1. Data Exploration <br/>
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

### 2. Pre-processing - Draft

To execute the experiments from a terminal, use the following command:
```python
python Loss_function.py && python Masking_Encoding.py && python Multiscale_Crossattention.py && python Tren_generation_module.py && python Uncertaintyaware_restoration.py
```

### Initial findings to replicate [ECGAD](https://github.com/MediaBrain-SJTU/ECGAD/tree/main): 
- No all `.mat` files contain `.hea` files.
- Content of the `val` key is presented as an array
```python
[[-127 -162 -197 ...  -18  -22  -21]]
```
- Multi-scale Cross-restoration was applied to a single signal `A00001.mat`.

## References
```bash
@inproceedings{jiang2022ecgad,
  title={Multi-scale Cross-restoration Framework for Electrocardiogram Anomaly Detection}
  author={Jiang, Aofan and Huang, Chaoqin and Cao, Qing and Wu, Shuang and Zeng, Zi and Chen, Kang and Zhang, Ya and Wang, Yanfeng},
  booktitle={International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI)},
  year={2023}
}
```

