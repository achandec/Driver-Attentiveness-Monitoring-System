#!/bin/bash
# Download dlib's facial landmark detector file to ./dlib_data 
cd dlib_data
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
cd ..
