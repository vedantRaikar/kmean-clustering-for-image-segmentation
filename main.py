#image segmentation using k mean method 

#import the required libraries 
import numpy as np 
import cv2 
import matplotlib.pyplot as plt 

#load the image in RGB 
original_image = cv2.imread(r"C:\Users\vedant raikar\Desktop\project\image clustering using k mean\test.jpeg")


#converting the image to hsv 
img = cv2.cvtColor(original_image , cv2.COLOR_BGR2HSV)

#This line of code is used to reshape a 2D or 3D image array to a 2D array of shape (n_pixels, 3) where n_pixels is the total number of pixels in the image.
#The variable img represents the input image array that needs to be reshaped. The .reshape() method is applied to img with the argument (-1, 3), which means that the number of rows in the output array will be automatically determined based on the size of the original array and the fact that each row in the output array will have 3 elements.
#By reshaping the image to this format, each row of the output array corresponds to a single pixel in the image, and the 3 columns represent the Red, Green, and Blue (RGB) color channels of that pixel. This is called a "vectorized" representation of the image because the image is now represented as a single long vector instead of a 2D or 3D array.
#This vectorized representation is often used as input to machine learning algorithms, which can use the RGB values of each pixel as features for image classification, object detection, or other tasks
vectorized = img.reshape((-1 , 3))


#The line of code vectorized = np.float32(vectorized) converts the data type of the vectorized array to 32-bit floating point precision using the NumPy library.
#The np.float32() function converts the data type of the input array to 32-bit floating point precision. This type of precision is commonly used in machine learning applications because it provides a good balance between accuracy and memory usage.
vectorized = np.float32(vectorized)



#cv2.TERM_CRITERIA_EPS indicates that the algorithm should terminate when the desired level of accuracy is achieved, while cv2.TERM_CRITERIA_MAX_ITER indicates that the algorithm should terminate when the maximum number of iterations is reached.
#The second value in the tuple, 10, specifies the maximum number of iterations that the algorithm should perform before terminating.
#The third value in the tuple, 1.0, specifies the desired level of accuracy that the algorithm should achieve before terminating.
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER , 10 , 1.0 )


#OpenCV provides cv2.kmeans(samples, nclusters(K), criteria, attempts, flags) function for color clustering.
K=3 
attempts = 10 
ret, label, center = cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)


#Now convert back into uint8.
center = np.uint8(center)

#we have to access the labels to regenerate the clustered image
res = center[label.flatten()]
result_image = res.reshape((img.shape))


#visualize the output result 
cv2.imshow('original image' , original_image)
cv2.imshow('resultant image ' , result_image)

cv2.waitKey(0)
cv2.destroyAllWindows()