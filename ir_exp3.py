import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

from math import pi, sin, cos, exp, sqrt, log

def box_muller(stdev, mean):
    """ Retrieve a normally distributed value by sampling the Box-Muller transform. """
    if not hasattr(box_muller, 'Z1'):
        # Box-Muller returns 2 values; only return one at a time.
        box_muller.Z1 = None
    elif box_muller.Z1:
        ret = box_muller.Z1 * stdev + mean
        box_muller.Z1 = None
        return ret

    U1, U2 = random.random(), random.random()
    Z0 = sqrt( -2 * log(U1) ) * cos(2 * pi * U2)
    box_muller.Z1 = sqrt( -2 * log(U1) ) * sin(2 * pi * U2)
   
    return Z0 * stdev + mean

# Experiment 3 (Image Restoration - motion blur)

# Apply motion blur to lenna
img = cv2.imread("images/lenna.png")
cv2.imshow('Original', img)
size = 30

# Generating motion blur kernel
kernel_motion_blur = np.zeros((size, size))
kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
kernel_motion_blur = kernel_motion_blur / size

# Apply kernel to the input image
output = cv2.filter2D(img, -1, kernel_motion_blur)

# Save motion blurred image
cv2.imwrite("images/motion_blur_lenna.png", output)


# Apply Gaussian noise to motion-blurred lenna
# with varying sigma values
for stdev in (1, 10, 100):
    img = cv2.imread("images/motion_blur_lenna.png")
    row,col,chn = img.shape
    mean = 0
    gauss = np.random.normal(mean, stdev,(row,col,chn))
    gauss = gauss.reshape(row,col,chn)
    noisy = img + gauss

    cv2.imwrite("images/Gaussian_blur_stdev_"+str(stdev)+"_motion_blur_lenna.png", noisy)
    
##############################################################################################
class MyImage:
    img = None
    name = ''

    def __init__(self,name):
        self.name = name
        self.img = cv2.imread(name)

def getH(u,v):
    a=b= 0.1
    T = 1
    H = 1
    muf = pi * ( (u*a) + (v*b) )
    if muf != 0.0:
        H = (T / muf) * sin(muf) * (cos(muf) + complex('j') * sin(muf))
    return H

# Apply Inverse Filtering to each degraded image
def inverse_filtering(image, r):
    # Open image and get file name
    img = cv2.imread(image)
    image_name = MyImage(image)
    fileName = image_name.name[image_name.name.find('/')+1:image_name.name.find('.')]
    
    # Create restored image
    restored_img = np.zeros(img.shape)

    # Inverse filtering algorithm
    # F_hat = G / H
    
    # Compute 2D FFT of F and G
    g = img[:,:]
    G = np.fft.fft2(g)
    F = np.fft.fft2(img)

    # Get H
    H = G / F
        
    # Inverse Filter
    F_hat = G / H
    
    # Rescale
    F_hat = F_hat*abs(G.max())
    
    # Apply inverse fft
    f_hat = np.fft.ifft2(F_hat)
    restored_img[:,:] = abs(f_hat)
    
    cv2.imwrite("images/Inverse_filtered_"+str(r)+"_"+str(fileName)+".png", restored_img)

for img in ("images/Gaussian_blur_stdev_1_motion_blur_lenna.png", 
            "images/Gaussian_blur_stdev_10_motion_blur_lenna.png", 
            "images/Gaussian_blur_stdev_100_motion_blur_lenna.png"):
    for r in (40, 70, 85):
        inverse_filtering(img, r)


# Apply Wiener Filtering to each degraded image
def wiener_filtering(image, k):
    img = cv2.imread(image)
    image_name = MyImage(image)
    fileName = image_name.name[image_name.name.find('/')+1:image_name.name.find('.')]
    restored_img = np.zeros(img.shape)
    
    # Wiener Filtering algorithm
    
    # Compute 2D FFT
    g = img[:,:]
    G = np.fft.fft2(g)
    F = np.fft.fft2(img)

    # Get H
    H = G / F

    # Find the wiener filter term
    weiner_term = (abs(H)**2)/(abs(H)**2 + k)
    H_weiner = (G/H)*weiner_term
                
    # Apply inverse FFT
    f_hat = np.fft.ifft2(H_weiner)
    restored_img[:,:] = abs(f_hat)

    cv2.imwrite("images/Wiener_filtered_"+str(k)+"_"+str(fileName)+".png", restored_img)

for img in ("images/Gaussian_blur_stdev_1_motion_blur_lenna.png", 
            "images/Gaussian_blur_stdev_10_motion_blur_lenna.png", 
            "images/Gaussian_blur_stdev_100_motion_blur_lenna.png"):
    for k in (1, 5, 8, 10, 20, 50, 100):
        wiener_filtering(img, k)