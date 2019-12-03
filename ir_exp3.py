import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

from math import pi, sin, cos, exp, sqrt, log
from PGM import PGMImage

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

def H(u,v,T,a,b):
    H = 1
    muf = pi * (u*a + v*b)
    if muf != 0.0: 
        H = (T / muf) * sin(muf) * (cos(muf) - sin(muf))
    return H


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
    

# TODO: Apply Inverse Filtering to each degraded image
def inverse_filtering(image, r):
    img = cv2.imread(image)
    restored_img = np.zeros(img.shape)

    # TODO: Inverse filtering algorithm
    # F_hat = G / H
    for i in range(0,3):
        # Compute 2D FFT
        g = img[:,:,i]
        G = np.fft.fft2(g)

        # Pad kernel with zeros
        h = np.zeros(g.shape)
        h_padded = np.zeros(g.shape)
        h_padded[:h.shape[0],:h.shape[1]] = np.copy(h)
        H = (np.fft.fft2(h_padded))

        # Normalize
        H_norm = H/abs(H.max())
        G_norm = G/abs(G.max())
        F_temp = G_norm/H_norm
        F_norm = F_temp/abs(F_temp.max())

        # Rescale
        F_hat = F_norm*abs(G.max())

        # Apply inverse fft
        f_hat = np.fft.ifft2(F_hat)
        restored_img[:,:,i] = abs(f_hat)

    cv2.imwrite("images/Inverse_filtered_"+str(r)+".png", restored_img)

''' 
for img in ("images/Gaussian_blur_stdev_1_motion_blur_lenna.png", 
            "images/Gaussian_blur_stdev_10_motion_blur_lenna.png", 
            "images/Gaussian_blur_stdev_100_motion_blur_lenna.png"):
    for r in (40, 70, 85):
        inverse_filtering(img, r)
'''
for r in (40, 70, 85):
    inverse_filtering("images/Gaussian_blur_stdev_1_motion_blur_lenna.png", r)


# TODO: Apply Wiener Filtering to each degraded image
def wiener_filtering(image, k):
    img = cv2.imread(image)
    restored_img = np.zeros(img.shape)
    
    # TODO: Wiener Filtering algorithm
    for i in range(0,3):
        # Compute 2D FFT
        g = img[:,:,i]
        G = np.fft.fft2(g)

        # Pad kernel with zeros
        h = np.zeros(g.shape)
        h_padded = np.zeros(g.shape)
        h_padded[:h.shape[0],:h.shape[1]] = np.copy(h)
        H = (np.fft.fft2(h_padded))

        # Find the inverse filter term
        weiner_term = (abs(H)**2 + k)/(abs(H)**2)
        print("max value of abs(H)**2 is ",(abs(H)**2).max())
        H_weiner = H*weiner_term
        
        # Normalize
        H_norm = H_weiner/abs(H_weiner.max())
        G_norm = G/abs(G.max())
        F_temp = G_norm/H_norm
        F_norm = F_temp/abs(F_temp.max())

        # Rescale
        F_hat  = F_norm*abs(G.max())
            
        # Apply inverse FFT
        f_hat = np.fft.ifft2(F_hat)
        restored_img[:,:,i] = abs(f_hat)

    cv2.imwrite("images/Wiener_filtered_"+str(k)+".png", restored_img)

'''
for img in ("images/Gaussian_blur_stdev_1_motion_blur_lenna.png", 
            "images/Gaussian_blur_stdev_10_motion_blur_lenna.png", 
            "images/Gaussian_blur_stdev_100_motion_blur_lenna.png"):
    for k in (1, 10, 20, 50, 100):
        wiener_filtering(img, k)
'''
for k in (1, 10, 20, 50, 100):
    wiener_filtering("images/Gaussian_blur_stdev_1_motion_blur_lenna.png", k)