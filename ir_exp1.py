import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy import fftpack

from Lib import spatially_filtered, Kernel
from PGM import PGMImage

# Experiment 1 (Noise Removal)

# Load boy_noisy image
im = plt.imread("images/boy_noisy.png").astype(float)
plt.figure()
plt.imshow(im, plt.cm.gray)
plt.title('Original image')

# Apply FFT to boy_noisy
im_fft = fftpack.fft2(im)
def plot_spectrum(im_fft):
    # Logarithmic colormap
    plt.imshow(np.abs(im_fft), norm=LogNorm(vmin=5))
    plt.colorbar()
plt.figure()
plot_spectrum(im_fft)
plt.title('Fourier transform')

# Define the fraction of coefficients (in each direction) we keep
keep_fraction = 1.0
# Copy fft and Set r and c to be the number of rows and columns of the array.
im_fft2 = im_fft.copy()
r, c = im_fft2.shape[0], im_fft2.shape[1] 
# Set to zero all rows and columns with indices between 
# r*keep_fraction and r*(1-keep_fraction):
im_fft2[int(r*keep_fraction):int(r*(1-keep_fraction))] = 0
im_fft2[:, int(c*keep_fraction):int(c*(1-keep_fraction))] = 0
plt.figure()
plot_spectrum(im_fft2)
plt.title('Filtered Spectrum')

# Apply inverse FFT
im_new = fftpack.ifft2(im_fft2).real
plt.figure()
plt.imshow(im_new, plt.cm.gray)
plt.title('Reconstructed Image')

plt.show()

# TODO: Extract noise pattern


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