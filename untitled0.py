# -*- coding: utf-8 -*-
"""
Created on Sat May 22 00:51:16 2021

@author: jkora
"""
import cv2
import numpy as np
import pandas as pd

# read image
img = cv2.imread('Black_n_White_hat_cleaned.png')
'''
#convert img to grey
img_grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#set a thresh
thresh = 100
#get threshold image
ret,thresh_img = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)
#find contours
contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

#create an empty image for contours
img_contours = np.zeros(img.shape)
# draw the contours on the empty image
cv2.drawContours(img_contours, contours, 2, (255,255,255), 3)
#save image
cv2.imwrite('T:/contours.png',img_contours)

#np.savetxt("hat_coord.csv", contours[2][:,0], delimiter=",")
print(contours[2][:,0])
# Initialize empty list
lst_intensities = []
lst_coord = []

# For each list of contour points...
for i in range(len(contours)):
    # Create a mask image that contains the contour filled in
    cimg = np.zeros_like(img)
    cv2.drawContours(cimg, contours, i, color=100, thickness=-1)

    # Access the image pixels and create a 1D numpy array then add to list
    pts = np.where(cimg == 100)
    lst_coord.append(pts[0:2])
    print(pts[0:2], '\n')
    lst_intensities.append(img[pts[0], pts[1]])


#contours = contours[0] if len(contours) == 2 else contours[1]
cntr = contours[2]

for pt in cntr:
    print(pt)

for i in range(len(contours)):

    indices = np.where(img_contours == 255)
    print(indices[0])
    coordinates = zip(indices[0], indices[1])
    print(coordinates)
#print(np.where(img_contours == 225))

'''
# convert to grayscale
bw = cv2.imread('Baseball_profile.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
print('b')
# threshold and invert so hexagon is white on black background
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
thresh = 255 - thresh
print('c')
# get contours
result = np.zeros_like(img)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#contours = contours[0] if len(contours) == 2 else contours[1]
cntr = contours[1]
#cv2.drawContours(result, [cntr], -1, (0,0,0), 1)
cv2.drawContours(result, contours, -1, (0,255,0), 3)

# print number of points along contour
print('number of points: ',len(cntr))

print('')

# list contour points
#for pt in cntr:
#    print(pt)
'''
# save resulting images
cv2.imwrite('blue_hexagon_thresh.png',thresh)
cv2.imwrite('blue_hexagon_contour.png',result)  
'''
# show thresh and contour   
#cv2.imshow("thresh", thresh)
#cv2.imshow("result", result)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

print('')

polygon = contours[1]

#def find_centroid(polygon):

hat_df = pd.DataFrame(data = polygon[:,0,0:2], columns = ['x','y'], dtype = int)    

#x_center = np.mean(polygon[:,:,0])
#y_center = np.mean(polygon[:,:,1])
centroid = (hat_df['x'].mean(), hat_df['y'].mean())

print(centroid)    

def PolyArea(x,y):
    return 0.5*np.abs(np.dot(np.roll(y,1),x)-np.dot(y,np.roll(x,1)))

print(PolyArea(polygon[:,0,0],polygon[:,0,1].T))

#hat_df.to_csv('Hat_outline.csv')