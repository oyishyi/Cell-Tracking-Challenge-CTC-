import cv2
import numpy as np
import pandas as pd
import math
    
class Cell:
    def __init__(self, center, area):
        self.centroid = center
        self.area = area
        self.path =[]

def preprocessing(frame):
    kernel = np.ones((4, 4), np.uint8)
    frame_erosion = cv2.erode(frame, kernel, iterations = 1)
    frame_dilation = cv2.dilate(frame_erosion, kernel, iterations = 1)
    
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(6, 6))
    gray = clahe.apply(frame_dilation)
    
    ret,thresh = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
    thresh = cv2.GaussianBlur(thresh, (3, 3), 0)# 高斯滤波
    
    return thresh

cap = cv2.VideoCapture('PhC-C2DL-PSC/test/t%03d.tif')
fps = 5    #保存视频的FPS，可以适当调整
size=(720,576)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
videoWriter = cv2.VideoWriter('output.avi',fourcc,fps,size)
first_frame = True
cells = []
lines = []


# Open the captures
while(cap.isOpened()):
    # read a capture
    success, frame = cap.read()
    if not success:
        break
    # Convert a image as grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # pre-processing
    thresh1 = preprocessing(gray)
    # find contours
    image, contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # First frame or not
    if first_frame:
        for contour in contours:
            M = cv2.moments(contour)
            # Centroid of contour
            centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            # Area of contour
            area = cv2.contourArea(contour)
            # Create cell's bounding boxes
            x, y, w, h = cv2.boundingRect(contour)
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
            # Create a cell
            cell = Cell(centroid, area)
            # Record the path
            cell.path.append(centroid)
            # Record this cell
            cells.append(cell)
            # Cell's bounding boxes
        first_frame = False
    else:
        for contour in contours:
            # A new cell or not
            new_cell = True
            M = cv2.moments(contour)
            # Centroid of contour
            centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            # Area of contour
            area = cv2.contourArea(contour)
            # Create cell's bounding boxes
            x, y, w, h = cv2.boundingRect(contour)
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
            for cell in cells:
                if math.sqrt(pow((cell.centroid[0] - centroid[0]), 2) + pow((cell.centroid[1] - centroid[1]), 2)) < 5:                       
                    if area > cell.area * 1.8:
                        x, y, w, h = cv2.boundingRect(contour)
                        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    if centroid != cell.path[-1]:
                        cell.path.append(centroid)
                        cell.centroid = centroid
                    new_cell = False
            if new_cell:
                cell = Cell(centroid, cv2.contourArea(contour))
                cell.path.append(centroid)
                cells.append(cell)
    for cell in cells:
        for i in range(len(cell.path) - 1):
            cv2.line(frame, cell.path[i], cell.path[i + 1], (255,0, 0), 1) 
    # Show the results
    cv2.imshow('frame', frame)
    videoWriter.write(frame)
    # A time interval of 100 milliseconds and input 'q' for quit
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
# Release everything if job is finished
cap.release()
videoWriter.release()
cv2.destroyAllWindows()