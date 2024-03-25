# ASCREM
## Anomaly Detection in Single Short ECG Recordings via Multi-scale Cross-Restoration Framework (ASCREM)

Author:Paul Yomer Ruiz Pinto (N10962646)
Last Modified: 25/03/2024<br/>
Code adapted from [ECGAD](https://github.com/MediaBrain-SJTU/ECGAD/tree/main) which is licensed under MIT License permitting modification and distribution.

--------------

## Project Abstract

In the realm of electrocardiogram (ECG) anomaly detection, where accurate identification of irregularities is crucial for effective clinical intervention, this study aims to replicate and extend the findings of a sophisticated multi-scale cross-restoration framework proposed previously. This framework utilizes a deep autoencoder architecture to detect abnormal ECG signals by integrating both local and global characteristics, mirroring the diagnostic process of experienced cardiologists. However, despite its potential, the model faces challenges in interpretability and sensitivity to outliers, limiting its clinical applicability and performance reliability. Addressing these gaps, our investigation focuses on refining anomaly detection criteria and enhancing preprocessing techniques to mitigate the impact of outliers on model performance. Specifically, we aim to refine raw ECG signals by noise elimination, introduce single-channel short ECG signals using autoencoders, and evaluate the proposed model's performance using various metrics. By comparing it with the existing model, we seek to contribute to the advancement of anomaly detection methodologies in ECG analysis, ultimately improving diagnostic accuracy and patient care. The source codes for the experiments are available at https://github.com/MediaBrain-SJTU/ECGAD/tree/main (existing model) and https://github.com/n10962646/ASCREM.git (study model, initial stage).  

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

