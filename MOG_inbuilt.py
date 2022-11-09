import cv2 as cv
from object_tracker import *

# Defining object tracker
tracker = Tracker()

# Defining Background Subtractor
bg_subtractor = cv.createBackgroundSubtractorMOG2(history=100, varThreshold=50)

# Taking video input for object tracking
cap = cv.VideoCapture("highway.mp4")

while True:
    val, frame = cap.read()

    # Obtaining Region of interest
    # roi = frame[340: 720,500: 800]
    roi = frame[200:600, 300:1000]

    # Detecting foreground object in the region of interest
    mask = bg_subtractor.apply(roi)
    _, mask = cv.threshold(mask, 200, 255, cv.THRESH_BINARY)

    # Applying contour to mask
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Creating an dictionary to store detected objects
    detection_dict = []

    for i in contours:
        # Calculate area and remove small elements
        area = cv.contourArea(i)
        # selecting possible moving objects based on the area change in terms of contours
        if area > 500:

            x, y, w, h = cv.boundingRect(i)

            detection_dict.append([x, y, w, h])

    # Applying the tracker to track the detected objects stored in dectection_dict
    object_ids = tracker.update(detection_dict)
    for object_id in object_ids:
        x, y, w, h, id = object_id

        # labelling the detected objects with tracker id and creating a green rectangle around the detected object
        cv.putText(roi, str(id), (x, y - 15), cv.FONT_ITALIC, 1, (0, 0, 0), 2)
        cv.rectangle(roi, (x, y), (x + w, y + h), (255, 0, 0), 3)

    cv.imshow("Object tracking", roi)
    cv.imshow("Mask", mask)
    # cv.imshow("Frame", frame)

    key = cv.waitKey(50)
    # Click 'space key' to stop the program
    if key == 32:
        break

cap.release()
cv.destroyAllWindows()
