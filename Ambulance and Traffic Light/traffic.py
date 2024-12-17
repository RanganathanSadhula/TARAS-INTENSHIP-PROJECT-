from cgitb import grey
from wsgiref import validate
import cv2
from cv2 import blur
from cv2 import dilate
import numpy as np

# web Camera
cap = cv2.VideoCapture('video2.mp4')

min_width_rectangle = 80
min_height_rectangle = 80

count_line_position = 550

# Initalize Subsstructor
algo = cv2.createBackgroundSubtractorMOG2(varThreshold=500)
object_detector = cv2.createBackgroundSubtractorMOG2()


def center_handle(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy


detect = []
offset = 6
counter = 0

while True:
    ret, frame1 = cap.read()

    roi = frame1[300:1280, 0:650]

    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (3, 3), 5)

    img_sub = algo.apply(blur)
    dilat = cv2.dilate(img_sub, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)
    counterShape, h = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(frame1, (50, count_line_position), (50, count_line_position), (225, 255, 102), 1)

    for (i, c) in enumerate(counterShape):
        (x, y, w, h) = cv2.boundingRect(c)
        validate_counter = (w >= min_width_rectangle) and (w >= min_height_rectangle)
        if not validate_counter:
            continue

        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 200, 0), 2)

        center = center_handle(x, y, w, h)
        detect.append(center)
        cv2.circle(frame1, center, 4, (0, 0, 200), -1)

        for (x, y) in detect:
            if y < (count_line_position + offset) and y > (count_line_position - offset):
                counter += 1

            cv2.line(roi, (50, count_line_position), (200, count_line_position), (225, 0, 0), 1)
            detect.remove((x, y))

            print("No.of Vehicle: " + str(counter))

    cv2.putText(frame1, "No.of Vehicle: " + str(counter), (450, 70), cv2.FONT_HERSHEY_TRIPLEX, 2, (0), 5)

    #cv2.imshow('Detecter', dilatada)
    cv2.imshow('video original', frame1)
    #cv2.imshow('roi', roi)

    if cv2.waitKey(1) == 13:
        break

cv2.destroyAllWindow()
cap.release()