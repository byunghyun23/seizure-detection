# A CNN-Attention Model for Epileptic Seizure Detection

## Introduction
This is a TensorFlow implementation for A CNN-Attention Model for Epileptic Seizure Detection.  
This work has been published in [IEEE Student Paper Contest 2021](https://github.com/byunghyun23/seizure-detection/blob/main/assets/program.pdf).  
Please refer to the [paper](https://github.com/byunghyun23/seizure-detection/blob/main/assets/paper.pdf) and the [prize](https://github.com/byunghyun23/seizure-detection/blob/main/assets/prize.pdf).

![image](https://github.com/byunghyun23/seizure-detection/blob/main/assets/fig1.PNG)

## Dataset
Before training a model, you need to download dataset [here](https://physionet.org/content/chbmit/1.0.0/).  
Then, move the downloaded dataset to
```
--data
```

## Preprocessing
Run
```
python preprocessing.py
```
After preprocessing, you can get preprocessed EEG data for each patient below.
```
--preprocess
```


## Train
```
python train.py
```
After training, you can get CNN-Attention models for each patient and performance of epileptic seizure detection below.
```
--model
--output
```
Patients with epileptic seizure duration of less than 30 seconds (chb06, chb16) were excluded from this paper.

![image](https://github.com/byunghyun23/seizure-detection/blob/main/assets/fig2.PNG)
