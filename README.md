# Driver-Attentiveness-Monitoring-System
A Computer Vision based approach to monitor a driver's attention to the road while driving.

-By Al Chandeck Chen

## Introduction
  A Driver Attentiveness Monitoring System (DAMS) system usually takes part within an autonomous vehicle (AV) and its task is to track the driver’s attention when the car is in autonomous mode. The idea is that a car’s self-driving capabilities are not infallible, and as such, it is necessary that the driver is attentive to the road and its surroundings in order to be able to react, in a timely manner, in case of failure of the autonomous driving system. Hence, this system main purpose is safety.
  
  Initial work in this system within Georgia Tech’s EcoCar team started last semester (Fall 2019). Previous members attempted to create a convolutional neural network (CNN) to inference if a driver was “paying attention” or doing other activities (e.g. talking to a passenger, texting, talking on the phone, and more). The CNN was trained on a dataset which was provided in a Kaggle competition and had over 200 thousand images for 10 different classes, among which, only one was the “paying attention class.”

  The scope of my work this semester (Spring 2020) was to incorporate a new approach to monitor driver’s attention using a computer vision techniques. My initial idea was to monitor facial features such as face pose, gaze, eye openness and detect facial landmarks, such as eyes, pupils, and so on, in order to ascertain if a driver was paying attention. My work also included researching state-of-the-art computer vision papers to accomplish this, setting up environment, and debugging both the code and hardware used for this project.

## Accomplishments
I have put together a doubly threaded application that enables us to…
With my work, I am able to successfully track real-time facial landmarks of the driver using Logitech C920 webcam connected to an NVIDIA Jetson AGX Xavier. I was also capable of tracking the gaze of the passenger relative to a coordinated system located at the camera’s lens. I also capable of detecting if the driver has fallen asleep by tracking the eye aspect ratio (EAR) over time.

## Future work
