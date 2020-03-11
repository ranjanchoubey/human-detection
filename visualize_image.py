import numpy as np 
import cv2,joblib
import Sliding as sd
from imutils.object_detection import non_max_suppression
import imutils
from skimage.feature import hog
from skimage import color
from skimage.transform import pyramid_gaussian


image = cv2.imread('test/test.jpg')
image = cv2.resize(image,(400,256))
min_wdw_sz = (64,128)
step_size = (9,9)
downscale = 1.25
#List to store the detections
detections = []
#The current scale of the image 
scale = 0
clf = joblib.load('models/models.dat')
for im_scaled in pyramid_gaussian(image, downscale = downscale):
    #The list contains detections at the current scale
    if im_scaled.shape[0] < min_wdw_sz[1] or im_scaled.shape[1] < min_wdw_sz[0]:
        break
    for (x, y, im_window) in sd.sliding_window(im_scaled, min_wdw_sz, step_size):
        if im_window.shape[0] != min_wdw_sz[1] or im_window.shape[1] != min_wdw_sz[0]:
            continue
        im_window = color.rgb2gray(im_window)
            
        fd=hog(im_window, orientations=9,pixels_per_cell=(8,8),visualize=False,cells_per_block=(3,3))
        fd = fd.reshape(1, -1)
        pred = clf.predict(fd)
     
        if pred == 1:
                
            if clf.decision_function(fd) > 0.5:
                detections.append((int(x * (downscale**scale)), int(y * (downscale**scale)), clf.decision_function(fd), 
                int(min_wdw_sz[0] * (downscale**scale)),
                int(min_wdw_sz[1] * (downscale**scale))))
                 

            
    scale += 1

clone = image.copy()

for (x_tl, y_tl, _, w, h) in detections:
    cv2.rectangle(image, (x_tl, y_tl), (x_tl + w, y_tl + h), (0, 255, 0), thickness = 2)

rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections])
sc = [score[0] for (x, y, score, w, h) in detections]
print ("sc: ", sc)
sc = np.array(sc)
pick = non_max_suppression(rects, probs = sc, overlapThresh = 0.3)
#print ("shape, ", pick.shape)

for(x1, y1, x2, y2) in pick:
    cv2.rectangle(clone, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(clone,'Person',(x1-2,y1-2),1,0.75,(121,12,34),1)
cv2.imshow('Person Detection',clone)
cv2.waitKey(0)
cv2.destroyAllWindows()