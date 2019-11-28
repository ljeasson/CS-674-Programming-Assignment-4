import math
import numpy as np
import matplotlib.pyplot as plt
import cv2

from PIL import Image
from Lib import spatially_filtered, Kernel
from PGM import PGMImage

# Experiment 1.a (Noise Removal)
img = cv2.imread("images/boy_noisy.png",0)

# Apply DFT and FFT shift
dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

# Get number of rows and columns,
# and center image
rows, cols = img.shape
center_row, center_col = rows/2 , cols/2

# Create mask
# Center square is 1, remaining all zeros
mask = np.zeros((rows,cols,2),np.uint8)
mask[int(center_row-30):int(center_row+30), int(center_col-30):int(center_col+30)] = 1

# Apply mask
fshift = dft_shift*mask

# Apply inverse DFT
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

# Plot images
plt.figure()
plt.imshow(img, plt.cm.gray)
plt.title('Original image')

plt.figure()
plt.imshow(magnitude_spectrum, plt.cm.gray)
plt.title('Magnitude Spectrum')

plt.figure()
plt.imshow(img_back, plt.cm.gray)
plt.title('Filtered image')

plt.show()

# TODO: Experiment 1.b (Extract noise pattern)

'''
# Apply Gaussian Smoothing (7x7) and (15x15)
p = PGMImage("images/boy_noisy.pgm")
gaussian_matrix_7 = Kernel(mask = [ [1,1,2,2,2,1,1], 
                                  [1,2,2,4,2,2,1], 
                                  [2,2,4,8,4,2,2], 
                                  [2,4,8,16,8,4,2], 
                                  [2,2,4,8,4,2,2], 
                                  [1,2,2,4,2,2,1], 
                                  [1,1,2,2,2,1,1] ])

gaussian_matrix_15 = Kernel(mask = [ [2,2,3,4,5,5,6,6,6,5,5,4,3,2,2],
                                     [2,3,4,5,7,7,8,8,8,7,7,5,4,3,2],
                                     [3,4,6,7,9,10,10,11,10,10,9,7,6,4,3],
                                     [4,5,7,9,10,12,13,13,13,12,10,9,7,5,4],
                                     [5,7,9,11,13,14,15,16,15,14,13,11,9,7,5],
                                     [5,7,10,12,14,16,17,18,17,16,14,12,10,7,5],
                                     [6,8,10,13,15,17,19,19,19,17,15,13,10,8,6],
                                     [6,8,11,13,16,18,19,20,19,18,16,13,11,8,6],
                                     [6,8,10,13,15,17,19,19,19,17,15,13,10,8,6],
                                     [5,7,10,12,14,16,17,18,17,16,14,12,10,7,5],
                                     [5,7,9,11,13,14,15,16,15,14,13,11,9,7,5],
                                     [4,5,7,9,10,12,13,13,13,12,10,9,7,5,4],
                                     [3,4,6,7,9,10,10,11,10,10,9,7,6,4,3],
                                     [2,3,4,5,7,7,8,8,8,7,7,5,4,3,2],
                                     [2,2,3,4,5,5,6,6,6,5,5,4,3,2,2] ])

p_gaussian = spatially_filtered(p, gaussian_matrix_7, normalize=True, truncate=False)
p_gaussian.save(f"smoothed-gaussian-7-{p.name}") 

p_gaussian = spatially_filtered(p, gaussian_matrix_15, normalize=True, truncate=False)
p_gaussian.save(f"smoothed-gaussian-15-{p.name}")
'''