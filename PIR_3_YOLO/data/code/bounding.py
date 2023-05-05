import numpy as np
import os
import cv2 as cv

file_path = '../BRATS/refined/labels/BRATS_001_e74.png'

im = cv.imread(file_path)
assert im is not None, "file could not be read, check with os.path.exists()"
imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(imgray, 0, 255, 3)

kernel = np.ones((5, 5), np.uint8)
thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)

contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

for cnt in contours:
   approx = cv.approxPolyDP(cnt, 0.01*cv.arcLength(cnt, True), True)
   (x,y)=cnt[0,0]

   if len(approx) >= 5:
        img2 = np.zeros(im.shape,np.uint8)
        print(approx)
        img2 = cv.drawContours(img2, [approx], -1, (0,255,0), 1)
        cv.imwrite("test poly.jpg",img2)

#full_name = os.path.basename(file_path)
#file_name = os.path.splitext(full_name)

#f = open(file_name[0]+".txt", "x")
#f.write(str(contours))
#f.close()