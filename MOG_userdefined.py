import numpy as np
import cv2 as cv
from object_tracker import *

# Defining object tracker
tracker = Tracker()

# Defining gaussian distribution pdf
def gauss_dis_pdf(x, mean, sigma):
    return (1/(np.sqrt(2*3.14)*sigma))*(np.exp(-0.5*(((x-mean)/sigma)**2)))

# Taking video input for object tracking
cap = cv.VideoCapture("highway.mp4")
_, frame = cap.read()

# Selecting Region of Interest for the object tracking
roi = frame[200:600, 300:1000]
# roi = frame[340: 720,500: 800]

roi = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)

# getting shape of the frame
row, col = roi.shape
# We will be using 3 mixtures of gaussian

# initialising mean,var,omega and omega by sigma
mean = np.zeros([3, row, col], np.float64)
mean[1, :, :] = roi

var = np.zeros([3, row, col], np.float64)
var[:, :, :] = 400

# Omega - weight associated with a gaussian
omega = np.zeros([3, row, col], np.float64)
omega[0, :, :], omega[1, :, :], omega[2, :, :] = 0, 0, 1

# Omega by sigma ratio
r = np.zeros([3, row, col], np.float64)

# initializing foreground and background
foreground = np.zeros([row, col], np.uint8)
background = np.zeros([row, col], np.uint8)

# initializing learning rate alpha and threshold T
alpha = 0.3
T = 0.5

# converting data type of integers 0 and 255 to uint8 type
a = np.uint8([255]) # White frame
b = np.uint8([0]) # Black frame

while cap.isOpened():
    _, frame = cap.read()
    # roi = frame[340: 720,500: 800]
    roi = frame[200:600, 300:1000]

    # Converting the roi into grayscale so that we gaussian distribution can be applied in 1D
    gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    gray = gray.astype(np.float64)

    # Because var becomes negative after some time because of gauss_dis_pdf function so we are converting those indices
    # values which are near zero to some higher values according to their preferences
    var[0][np.where(var[0] < 1)] = 10
    var[1][np.where(var[1] < 1)] = 5
    var[2][np.where(var[2] < 1)] = 1

    # calulating standard deviation
    sigma1 = np.sqrt(var[0])
    sigma2 = np.sqrt(var[1])
    sigma3 = np.sqrt(var[2])

    # getting values for the inequality test to get indexes of fitting indexes
    compare_val_1 = cv.absdiff(gray, mean[0])
    compare_val_2 = cv.absdiff(gray, mean[1])
    compare_val_3 = cv.absdiff(gray, mean[2])

    value1 = 2.5 * sigma1
    value2 = 2.5 * sigma2
    value3 = 2.5 * sigma3

    # Finding those indices where values of T are less than most probable gaussian and those where sum of most probale
    # and medium probable is greater than T and most probable is less than T
    fore_index1 = np.where(omega[2] > T)
    fore_index2 = np.where(((omega[2]+omega[1]) > T) & (omega[2] < T))

    # Finding those indices where a particular pixel values fits at least one of the gaussian
    gauss_fit_index1 = np.where(compare_val_1 <= value1)
    gauss_not_fit_index1 = np.where(compare_val_1 > value1)

    gauss_fit_index2 = np.where(compare_val_2 <= value2)
    gauss_not_fit_index2 = np.where(compare_val_2 > value2)

    gauss_fit_index3 = np.where(compare_val_3 <= value3)
    gauss_not_fit_index3 = np.where(compare_val_3 > value3)

    temp = np.zeros([row, col])
    temp[fore_index1] = 1
    temp[gauss_fit_index3] = temp[gauss_fit_index3] + 1
    index3 = np.where(temp == 2)

    temp = np.zeros([row, col])
    temp[fore_index2] = 1
    index = np.where((compare_val_3 <= value3) | (compare_val_2 <= value2))
    temp[index] = temp[index]+1
    index2 = np.where(temp == 2)

    match_index = np.zeros([row, col])
    match_index[gauss_fit_index1] = 1
    match_index[gauss_fit_index2] = 1
    match_index[gauss_fit_index3] = 1
    not_match_index = np.where(match_index == 0)

    # Updating variance and mean value of the matched indices of all three gaussians
    # Gaussian1
    rho = alpha * gauss_dis_pdf(gray[gauss_fit_index1],
                                mean[0][gauss_fit_index1], sigma1[gauss_fit_index1])
    constant = rho * ((gray[gauss_fit_index1] -
                      mean[0][gauss_fit_index1]) ** 2)
    mean[0][gauss_fit_index1] = (
        1 - rho) * mean[0][gauss_fit_index1] + rho * gray[gauss_fit_index1]
    var[0][gauss_fit_index1] = (1 - rho) * var[0][gauss_fit_index1] + constant
    omega[0][gauss_fit_index1] = (
        1 - alpha) * omega[0][gauss_fit_index1] + alpha
    omega[0][gauss_not_fit_index1] = (
        1 - alpha) * omega[0][gauss_not_fit_index1]

    # Gaussian2
    rho = alpha * gauss_dis_pdf(gray[gauss_fit_index2],
                                mean[1][gauss_fit_index2], sigma2[gauss_fit_index2])
    constant = rho * ((gray[gauss_fit_index2] -
                      mean[1][gauss_fit_index2]) ** 2)
    mean[1][gauss_fit_index2] = (
        1 - rho) * mean[1][gauss_fit_index2] + rho * gray[gauss_fit_index2]
    var[1][gauss_fit_index2] = (
        1 - rho) * var[1][gauss_fit_index2] + rho * constant
    omega[1][gauss_fit_index2] = (
        1 - alpha) * omega[1][gauss_fit_index2] + alpha
    omega[1][gauss_not_fit_index2] = (
        1 - alpha) * omega[1][gauss_not_fit_index2]

    # Gaussian3
    rho = alpha * gauss_dis_pdf(gray[gauss_fit_index3],
                                mean[2][gauss_fit_index3], sigma3[gauss_fit_index3])
    constant = rho * ((gray[gauss_fit_index3] -
                      mean[2][gauss_fit_index3]) ** 2)
    mean[2][gauss_fit_index3] = (
        1 - rho) * mean[2][gauss_fit_index3] + rho * gray[gauss_fit_index3]
    var[2][gauss_fit_index3] = (1 - rho) * var[2][gauss_fit_index3] + constant
    omega[2][gauss_fit_index3] = (
        1 - alpha) * omega[2][gauss_fit_index3] + alpha
    omega[2][gauss_not_fit_index3] = (
        1 - alpha) * omega[2][gauss_not_fit_index3]

    # Updating least probable gaussian for those pixel values which do not match any of the gaussian
    mean[0][not_match_index] = gray[not_match_index]
    var[0][not_match_index] = 200
    omega[0][not_match_index] = 0.1

    # Normalizing omega
    sum = np.sum(omega, axis=0)
    omega = omega/sum

    # Finding omega by sigma ratio for getting the background and foreground
    r[0] = omega[0] / sigma1
    r[1] = omega[1] / sigma2
    r[2] = omega[2] / sigma3

    # getting index order for sorting omega by sigma
    index = np.argsort(r, axis=0)

    # from that index(line 139) sorting mean,var and omega
    mean = np.take_along_axis(mean, index, axis=0)
    var = np.take_along_axis(var, index, axis=0)
    omega = np.take_along_axis(omega, index, axis=0)

    gray = gray.astype(np.uint8)

    # Getting background
    background[index2] = gray[index2]
    background[index3] = gray[index3]

    # Object Detection
    mask = cv.subtract(gray, background)
    _, mask = cv.threshold(mask, 50, 255, cv.THRESH_BINARY)

    # Applying contour to mask
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    detection_dict = []

    for i in contours:
        # Calculate area and remove small elements
        area = cv.contourArea(i)
        # selecting possible moving objects based on the area change in terms of contours
        if area > 120:

            x, y, w, h = cv.boundingRect(i)

            detection_dict.append([x, y, w, h])

    object_ids = tracker.update(detection_dict)
    for object_id in object_ids:
        x, y, w, h, id = object_id
        cv.putText(roi, str(id), (x, y - 15), cv.FONT_ITALIC, 1, (0, 0, 0), 2)
        cv.rectangle(roi, (x, y), (x + w, y + h), (255, 0, 0), 3)

    cv.imshow('Object tracking', roi)
    cv.imshow('Mask',mask)
    # cv.imshow("Frame", frame)

    key = cv.waitKey(50)
    # Click 'space key' to stop the program
    if key == 32:
        break

cap.release()
cv.destroyAllWindows()
