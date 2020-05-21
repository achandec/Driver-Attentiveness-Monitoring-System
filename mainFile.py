#!/usr/bin/env python

# load the library using the import keyword
# OpenCV must be properly installed for this to work. If not, then the module will not load with an
# error message.


# In Al's Windows computer: Open miniconda terminal and activate 'vip' virtual environment by typing: "conda activate vip"
# In NVIDIA: 
	# 1. open terminal 
	# 2.$ workon vip
	# 3.$ cd ~/Projects/DAMS/HMI_Thingy_SP2020
import cv2
import sys
import dlib
import csv
import warnings
import numpy as np
import pandas as pd
from imutils import face_utils
from scipy.spatial import distance as dist
from myVideoUtils import bb2rect
from myVideoUtils import getEyesAndEARs # EAR: Eye-Aspect-Ratio
from myVideoUtils import drawEye
from myVideoUtils import get_head_pose
from scipy.ndimage import gaussian_filter


print("[INFO] Initializing")

# Used to draw the 3D box around face to aid head pose visualization
line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
              [4, 5], [5, 6], [6, 7], [7, 4],
              [0, 4], [1, 5], [2, 6], [3, 7]]

# Used to Output EAR values to a file
fieldnames = ["frame_number", "leftEAR", "rightEAR", "Rx", "Ry", "Rz"]
with open('GlintingEARs.csv', 'w') as csv_file:
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()

# Creates cv2's Face Classifier and 
# dlib's Facial Landmark Predictor
cascPath = "./haar_data/haarcascade_frontalface_default.xml"
predictorPath = "./dlib_data/shape_predictor_68_face_landmarks.dat"
faceCascade = cv2.CascadeClassifier(cascPath)
predictor = dlib.shape_predictor(predictorPath)

#Used to get EAR dynamic threshold (EAR_Threshold_Function) to detect closed eyes
# Import data stored in calibration file
data = pd.read_csv('./calibration_files/EARvsRxyz.csv')
frame_number_calibration = data['frame_number']
leftEAR_calibration = data['leftEAR']
rightEAR_calibration = data['rightEAR']
Rx_calibration = data['Rx']
Ry_calibration = data['Ry']
Rz_calibration = data['Rz']
# Preprocess the data (filter)
leftEAR_calibration = leftEAR_calibration.rolling(5).mean() #rolling window avg (size=5) 
rightEAR_calibration = rightEAR_calibration.rolling(5).mean()
Rx_calibration = Rx_calibration.rolling(5).mean()
leftEAR_calibration = leftEAR_calibration.dropna() #drop NaN
rightEAR_calibration = rightEAR_calibration.dropna()
Rx_calibration = Rx_calibration.dropna()
leftEAR3 = gaussian_filter(leftEAR_calibration, sigma=3, order=0)# Gaussian filter sigma=3
#Fit a polynomial and generate a function called p30
with warnings.catch_warnings():
    warnings.simplefilter('ignore', np.RankWarning)
    EAR_Threshold_Function = np.poly1d(np.polyfit(Rx_calibration, leftEAR3, 30))


### Helper Functions ###
def detectDriverFace(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # The face or faces in an image are detected
    # This section requires the most adjustments to get accuracy on face being detected.
    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(1,1),flags = cv2.CASCADE_SCALE_IMAGE)
    
    # Print number of faces detected
    #print("Detected {0} faces!".format(len(faces)))
    
    # Extract the biggest face/boundingBox
    
    maxArea = 0 
    box=[0,0,0,0] 
    for (x, y, w, h) in faces:
        if(w*h > maxArea):
            maxArea = w*h
            box = [x,y,w,h]
    # Draw green box around the largest face detected
    cv2.rectangle(image, (box[0],box[1]), (box[0]+box[2], box[1]+box[3]), (0, 255, 0), 2)
	
    # Convert bounding box format to dlib's rectangle format and
    # Detect the facial landmarks with dlib's shape predictor
    dlibRect = bb2rect(box)
    landmarks = predictor(gray, dlibRect)
    landmarks = face_utils.shape_to_np(landmarks)
    
    # loop over the (x, y)-coordinates for the facial landmarks
    # and draw them on the image
    for (x, y) in landmarks:
    	cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

    return box, landmarks

	
def streamVideo(graySc = False, camUsed = 0, mode = "stream", vidFileName = ''):
    # Modify these parameters before running
    GrayScale = graySc # Change to True if you want video in graySacle
    CameraUsed = camUsed  # If computer as no built-in camera, write 0 to use an external webcam.
                          # If computer has built-in camera, write 1 to use an external webcam, or 0 to use built-in cam 
    StopStreaming = False # Change to 1 by pressing s if you want to stop streaming
    RecordEARs = False # Change to 1 by pressing r if you want to start outputting EARS to data.csv
    prevState = False # Keeps track of previous RecordEARs state (0/1)
    if mode == "stream":
        # Open stream of video at 30fps:
        cap = cv2.VideoCapture(CameraUsed, cv2.CAP_V4L)
        cap.set(cv2.CAP_PROP_FPS, 10) # set fps to 2,5,10,15,or 30
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        # Set resolution to 480p (no need for higher resolution)
        cap.set(3, 640)
        cap.set(4, 480)
        print("Camera Frame rate set at: ", frame_rate)
        
    elif mode == "file":
        # Playing video from file:
        cap = cv2.VideoCapture(vidFileName)
    # define constant for the number of consecutive
    # frames the eye must be below the threshold to be
    # considered closed
    CONSEC_FRAMES_FOR_BLINK = 3
    CONSEC_FRAMES_ASLEEP_ALERT = 1*frame_rate # ~1 sec 

    # initialize the frame counters and the total number of times
    # the driver has fallen asleep    
    COUNTER = 0
    TOTAL = 0
    currentFrame = 0
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('ClosedEyesDemo.avi',fourcc, 15.0, (640,480))
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Saves image of the current frame in jpg file
        # name = 'frame' + str(currentFrame) + '.jpg'
        # cv2.imwrite(name, frame)
        
        # Get driver's face bounding-box, landmarks, headpose, EARs, 
        # draw eyes, and extract head pose's Euler angles (Rx,Ry,Rz)
        box,landmarks = detectDriverFace(frame)
        reprojectdst, euler_angle = get_head_pose(landmarks)
        leftEAR, rightEAR, leftEye, rightEye = getEyesAndEARs(frame, landmarks)
        drawEye(frame, leftEye)
        drawEye(frame, rightEye)
        Rx = -euler_angle[0, 0] #negate so positive is when driver looks upward
        Ry = euler_angle[1, 0]
        Rz = euler_angle[2, 0]

        if RecordEARs and StopStreaming == False:
            if prevState == False:
                startingFrame = currentFrame
                prevState = True
            frame_number = currentFrame - startingFrame
            with open('GlintingEARs.csv', 'a') as csv_file:
                csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                info = {
                "frame_number": frame_number,
                "leftEAR": leftEAR,
                "rightEAR": rightEAR,
	        "Rx": Rx,
                "Ry": Ry,
                "Rz": Rz
                }
                csv_writer.writerow(info)
            print("Recording EARs")
        elif RecordEARs == False and prevState == True:
            prevState = False
            print("Stopped Recording EARs")

	# Draw 3D box around the face to visualize pose
        for start, end in line_pairs:
            cv2.line(frame, reprojectdst[start], reprojectdst[end], (0, 0, 255))

	# Mirror current frame and anotate
        frame = cv2.flip(frame,1)
        cv2.putText(frame, "Rx: " + "{:7.2f}".format(Rx), (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                         0.75, (0, 0, 0), thickness=2)
        cv2.putText(frame, "Ry: " + "{:7.2f}".format(Ry), (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                         0.75, (0, 0, 0), thickness=2)
        cv2.putText(frame, "Rz: " + "{:7.2f}".format(Rz), (20, 110), cv2.FONT_HERSHEY_SIMPLEX,
                         0.75, (0, 0, 0), thickness=2)

	# check to see if the eye aspect ratio is below the blink
	# threshold, and if so, increment the blink frame counter
        if Rx < 30 and leftEAR < EAR_Threshold_Function(Rx):
            COUNTER += 1
            if COUNTER >= CONSEC_FRAMES_ASLEEP_ALERT:
                cv2.putText(frame, "ASLEEP ALERT!", (250, 50),
		cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            if COUNTER >= CONSEC_FRAMES_FOR_BLINK:
                cv2.putText(frame, "EYE: CLOSED!", (240, 30),
		    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
	# otherwise, the eye aspect ratio is not below the blink
	# threshold
        elif Rx < 30 and leftEAR >= EAR_Threshold_Function(Rx):
        # if the eyes were closed for a sufficient number of
        # then increment the total number of blinks
            if COUNTER >= CONSEC_FRAMES_ASLEEP_ALERT:
                TOTAL += 1
        # reset the eye frame counter
            COUNTER = 0
         #   cv2.putText(frame, "EYE: OPEN", (240, 30),
	#	cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        elif Rx>=30 and leftEAR < 0.20:
            COUNTER += 1
            if COUNTER >= CONSEC_FRAMES_ASLEEP_ALERT:
                cv2.putText(frame, "ASLEEP ALERT!", (250, 50),
		cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            if COUNTER >= CONSEC_FRAMES_FOR_BLINK:
                cv2.putText(frame, "EYE: CLOSED!", (240, 30),
		    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
	# otherwise, the eye aspect ratio is not below the blink
	# threshold
        elif Rx >=30 and leftEAR > 0.20:
        # if the eyes were closed for a sufficient number of
        # then increment the total number of blinks
            if COUNTER >= CONSEC_FRAMES_ASLEEP_ALERT:
                TOTAL += 1
        # reset the eye frame counter
            COUNTER = 0
         #   cv2.putText(frame, "EYE: OPEN", (240, 30),
	#	cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            
	
	# draw the total number of blinks on the frame along with
	# the computed eye aspect ratio for the frame
        cv2.putText(frame, "ASLEEP: {}".format(TOTAL), (10, 30),
		cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.putText(frame, "leftEAR: {:.2f}".format(leftEAR), (480, 30),
		cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        
        if GrayScale and StopStreaming == False:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # write the flipped frame
            #out.write(img)
            cv2.imshow('frame',gray)
            out.write(gray)
        elif GrayScale == False and StopStreaming == False:
            # write the flipped frame
            #out.write(frame)
            #img = cv2.resize(img, (400,400), interpolation = cv2.INTER_AREA)
            cv2.imshow('frame',frame)
            out.write(frame)
        
        # Check for key-press and trigger corresponding interrupts
        k = cv2.waitKey(33)
        if k == 27 or k == ord('q'): 
            #if ESC or 'q' is pressed
            break
        elif k == ord('g'):
            # if 'g' key is pressed
            # toggle grayScale display
            GrayScale = ~GrayScale
        elif k == ord('r'):
            # if 'r' key is pressed
            # toggle RecordEARs 
            RecordEARs = ~RecordEARs
        elif k == ord('s'):
            # if 'g' key is pressed
            # toggle StopStreaming 
            StopStreaming = ~StopStreaming

            

        # To stop duplicate images
        currentFrame += 1

    # When everything done, release the capture
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    #  !!!!!!!!!!!! !!!!!!!!!!!! !!!!!!!!!!!! !!!!!!!!!!!! !!!!!!!!!!!! !!!!!!!!!!!! 
    # !!!!!!!!!!!! Modify these parameters before running !!!!!!!!!!!!!!!!!!!!!!!!!!!
    #  !!!!!!!!!!!! !!!!!!!!!!!! !!!!!!!!!!!! !!!!!!!!!!!! !!!!!!!!!!!! !!!!!!!!!!!! 
    GrayScale = False # Change to True if you want video in graySacle
    CameraUsed = 0  # If computer as no built-in camera, write 0 to use an external webcam.
                    # If computer has built-in camera, write 1 to use an external webcam, or 0 to use built-in cam 
    VideoFileorStream = 0 # 0 for Video Feed, 1 for Video File, 2 for single Image File (no video)

    if VideoFileorStream == 0:
        streamVideo(GrayScale, CameraUsed, mode="stream")

    elif VideoFileorStream == 1:
        # Gets the name of the video file (vidName) from sys.argv
        vidName = sys.argv[1]
        streamVideo(GrayScale, CameraUsed, mode="file", vidFileName=vidName)

    elif VideoFileorStream == 2:
        # Gets the name of the image file (imagePath) from sys.argv
        imagePath = sys.argv[1]
        image = cv2.imread(imagePath)
        img = detectFace(image)
        cv2.imshow("Faces Detected", image)
        cv2.waitKey(0) # Displays the image until any key is pressed
