import math
import numpy as np
import matplotlib.pyplot as plt
import cv2


# Experiment 1.a (Noise Removal)
img = cv2.imread("images/boy_noisy.png",0)

# Apply DFT and FFT shift
dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

# Create magnitude spectrum
magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

# Get number of rows and columns, and center of image
rows, cols = img.shape
center_row, center_col = rows/2 , cols/2

# Create rect mask
# Center square is 1, remaining all zeros
mask = np.zeros((rows,cols,2),np.uint8)
mask[int(center_row-30):int(center_row+30), int(center_col-30):int(center_col+30)] = 1

# Apply mask
fshift = dft_shift*mask

# Apply inverse DFT
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])


# Experiment 1.b (Extract noise pattern)

# Create inverted rect mask
mask_inv = np.ones((rows,cols),np.uint8)
mask_inv[int(center_row-30):int(center_row+30), int(center_col-30):int(center_col+30)] = 0

# Extract noise pattern
noise_pattern = magnitude_spectrum*mask_inv
ifft_noise_pattern = np.fft.ifft2(noise_pattern)
ifft_shift_noise_pattern = np.fft.fftshift(np.log(ifft_noise_pattern))

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

plt.figure()
plt.imshow(ifft_shift_noise_pattern.astype(float), plt.cm.gray)
plt.title('Noise Pattern')

plt.show()