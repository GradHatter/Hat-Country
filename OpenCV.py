# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 02:42:18 2020

@author: jkora
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('Baseball_profile.jpg')

mask = np.zeros(img.shape[:2],np.uint8)

bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

rect = (820,650,1400,2850)
cv2.grabCut(img,mask,rect,bgdModel,fgdModel,1,cv2.GC_INIT_WITH_RECT)
print(1)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]

gwashBW = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #change to grayscale
gwashBW = np.where(gwashBW == 0, 255, gwashBW)
print(2)
ret,thresh1 = cv2.threshold(gwashBW,150,254,cv2.THRESH_BINARY)
kernel = np.ones((5,5),np.uint8) #square image kernel used for erosion
erosion = cv2.erode(thresh1, kernel,iterations = 1) #refines all edges in the binary image
print(3)
opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel) #this is for further removing small noises and holes in the image
print(4)
'''
plt.imshow(closing, 'gray') #Figure 2
plt.xticks([]), plt.yticks([])
plt.show()
'''
contours, hierarchy = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #find contours with simple approximation
print(5)
#cv2.imshow('cleaner', closing) #Figure 3
print(6)
cv2.drawContours(closing, contours, -1, (255, 255, 255), 4)
#cv2.waitKey(0)

areas = [] #list to hold all areas


for contour in contours:
  #plt.imshow(cv2.drawContours(closing, contour, 0, (255, 255, 255), 3, maxLevel = 0))
  ar = cv2.contourArea(contour)
  areas.append(ar)

print(contours[6])  
print(areas)
print(7)
'''
max_area = max(areas)
max_area_index = areas.index(max_area) #index of the list element with largest area
# max area is just a rectangle so this doesn't work
'''
#contours = contours[1:5]
closing = cv2.drawContours(closing, contours, 6, (255, 255, 255), 3, maxLevel = 0)
plt.imshow(closing)
#cv2.imshow('cleaner', closing)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
