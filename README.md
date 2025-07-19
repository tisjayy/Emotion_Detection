# ResNet-Based Real-Time Emotion Detection

This project implements a ResNet-based Convolutional Neural Network (CNN) to classify human emotions from facial images in real time. The model is trained on the FER2013 dataset and integrated with OpenCV to capture live webcam input.

## Features
- Trains a custom ResNet-based CNN with L2 regularization and batch normalization.
- Achieves **75% test accuracy** on FER2013 after 40 epochs.
- Real-time emotion recognition with webcam using OpenCV.
- Uses **data augmentation** and experiment tracking with **MLflow**.

## Emotion Classes
The model classifies facial expressions into **7 emotion categories**:

- Angry  
- Disgust  
- Fear  
- Happy  
- Sad  
- Surprise  
- Neutral  

 

## Dataset
The project uses the [FER2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013).  
Training and testing data are stored as CSV files (`train.csv` and `test.csv`).

## Requirements
- use python 3.10 and set up a virtual environment.

Install the dependencies:
```bash
pip install numpy pandas tensorflow keras opencv-python scikit-learn mlflow
python emotion_recognition_final.py 
python face_tracking.py

