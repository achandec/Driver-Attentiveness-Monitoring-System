import numpy as np 
import cv2
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist

# The following are needed head pose estimation
K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
     0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
     0.0, 0.0, 1.0]
D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]

cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)

object_pts = np.float32([[6.825897, 6.760612, 4.402142],
                         [1.330353, 7.122144, 6.903745],
                         [-1.330353, 7.122144, 6.903745],
                         [-6.825897, 6.760612, 4.402142],
                         [5.311432, 5.485328, 3.987654],
                         [1.789930, 5.393625, 4.413414],
                         [-1.789930, 5.393625, 4.413414],
                         [-5.311432, 5.485328, 3.987654],
                         [2.005628, 1.409845, 6.165652],
                         [-2.005628, 1.409845, 6.165652],
                         [2.774015, -2.080775, 5.048531],
                         [-2.774015, -2.080775, 5.048531],
                         [0.000000, -3.116408, 6.097667],
                         [0.000000, -7.415691, 4.070434]])

reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                           [10.0, 10.0, -10.0],
                           [10.0, -10.0, -10.0],
                           [10.0, -10.0, 10.0],
                           [-10.0, 10.0, 10.0],
                           [-10.0, 10.0, -10.0],
                           [-10.0, -10.0, -10.0],
                           [-10.0, -10.0, 10.0]])

# Grab indexes of the facial landmarks for both eyes
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

#EAR: eye aspect ration
def eyeAspectRatio(eye):
    # Vertical distances of the corresponding eye landmark points
    V1 = dist.euclidean(eye[1], eye[5])
    V2 = dist.euclidean(eye[2], eye[4])
    # Horizontal distance of corresponding eye landmark points
    Horizontal = dist.euclidean(eye[0], eye[3])
    # Compute the eye aspect ratio
    EAR = (V1 + V2) / (2.0 * Horizontal)
    return EAR

def getEyesAndEARs(img, landmarks):
    # Extract the left and right eye coordinates
    leftEye = landmarks[lStart:lEnd]
    rightEye = landmarks[rStart:rEnd]
    # Use coordinates to compute the eye aspect ratio for both eyes
    leftEAR = eyeAspectRatio(leftEye)
    rightEAR = eyeAspectRatio(rightEye)
    
    return leftEAR, rightEAR, leftEye, rightEye
	
def drawEye(img, eyeLandmarks):
    # Compute the convex hull for the Eye
    eyeHull = cv2.convexHull(eyeLandmarks)
    # Draw the Eye on image
    cv2.drawContours(img, [eyeLandmarks], -1, (0, 255, 0), 1)

#This function converts a bounding box
def bb2rect(box):
    left = box[0]
    top = box[1]
    right = box[0]+box[2]
    bottom = box[1]+box[3]
    dlibRect = dlib.rectangle(left, top, right, bottom)
    return dlibRect

def get_head_pose(shape):
    image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                            shape[39], shape[42], shape[45], shape[31], shape[35],
                            shape[48], shape[54], shape[57], shape[8]])

    _, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs)

    reprojectdst, _ = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix,
                                        dist_coeffs)

    reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))

    # calc euler angle
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

    return reprojectdst, euler_angle


