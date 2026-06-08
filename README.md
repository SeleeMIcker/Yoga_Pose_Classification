# Yoga Pose Classification #

## Overview

Real-time yoga pose classification using a webcam, MediaPipe, and Trained Keras Model. 

This project is using MediaPipe extract the Yoga pose landmark (33 Body LandMark) in XYZ values to train the Model. It enables the model to classify User Yoga Pose in real time. 

## Dataset
The dataset includes 5 poses,
Downdog, Goddess, Plank, Tree and Warrior2. The Dataset is from kaggle.

## Link for Dataset Used
https://www.kaggle.com/datasets/niharika41298/yoga-poses-dataset 
-----------------------------------------------
## How it works
1. Webcam frame
2. MediaPipe 33 Body Landmark
3. 33 Landmark flatten to 132-element feature vector
4. Keras classifier, find the probability per pose
5. Smoothing (rolling average over last 10 frames)
6. Overlay Label + confidence bar on live feed
-----------------------------------------------
## Project Files
1. yoga_pose_webcam.py : Main script
2. yoga_pose_classifier.h5 : Trained Keras Model
3. Label_encoder.pk1 : Sklearn LabelEncoder, mapping index --> class
4. pose_landmarker.task MediaPipe Landmark model bundle 
5. open_camera.py: To check the camera status

### Download the model trained inside the google Colab, and copy the file into the YOGA_POSE folder
1. label_encoder.pkl
2. yoga_pose_classifier.h5

The "pose_landmarker.task" is download from online (compulsory)
// wget -O pose_landmarker.task -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task
-----------------------------------------------
1. tensorflow==2.20.0
2. keras==3.12.2
Need to ensure these packages version consistency with Google Colab version used for training
------------------------------------------------
## Set Up
1. cd /home/genai/Desktop/yoga_pose/ 
2. source venv/bin/activate
3. python3 yoga_pose_webcam.py
-------------------------------------------------
