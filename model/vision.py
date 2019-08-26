import cv2
import numpy as np

def get_threshold(gray):
    thresh = cv2.adaptiveThreshold(gray.reshape(224,224),
           255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    return thresh

def get_unknown(thresh):
    closingKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, closingKernel)
    dist_transform = cv2.distanceTransform(closed, cv2.DIST_L2, 3)
    _, fg = cv2.threshold(dist_transform,dist_transform.max(), 255, 0)
    fg = np.uint8(fg)
    return fg, cv2.subtract(thresh, fg)
    
def get_marker(fg, unknown, num_pixels=216):
    ret, marker = cv2.connectedComponents(fg)
    marker[unknown==255] = 127
    marker[unknown==0] = -127
    return marker

def get_marker_for_optimization(fg, unknown):
    ret, marker = cv2.connectedComponents(fg)
    marker[unknown==255] = 1
    return marker

