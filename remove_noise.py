import math
import numpy as np
import matplotlib.pyplot as plt

from Lib import spatially_filtered, Kernel
from PGM import PGMImage

def my_cfft(arr, inverse=False):
    # from ipdb import set_trace; set_trace()
    N = len(arr)
    is_power_two = lambda n: n & (n - 1) == 0
    assert is_power_two(N)
    
    import ctypes
    
    # Pad with leading zero and zeroes for complex
    ArrT = ctypes.c_float * (N * 2 + 1)
    c_arr = [0] * (N * 2 + 1)
    for i in range(N):
        if type(arr[i]) == complex:
            c_arr[(i * 2) + 1] = arr[i].real
            c_arr[(i * 2) + 2] = arr[i].imag
        else:
            c_arr[(i * 2) + 1] = arr[i]
    c_arr = ArrT(*c_arr)
    
    # Run 1-D Fourier Transform from fft.c
    cfft = ctypes.cdll.LoadLibrary('./fft.so')
    cfft.fft(ctypes.byref(c_arr), N, -1 if not inverse else 1)
    
    # Convert back to Python list
    div = N if inverse else 1
    return [complex(c_arr[(i * 2) + 1], c_arr[(i * 2) + 2]) / div for i in range(N)]

def cfft2(arr2, inverse=False):
    fft_out = [my_cfft(row, inverse) for row in arr2]
    
    cols = len(arr2[0])
    for i in range(cols):
        col = [row[i] for row in fft_out]
        col = my_cfft(col, inverse)
        
        for j in range(len(arr2)):
            fft_out[j][i] = col[j]
    
    return fft_out

# Experiment 1 (Noise Removal)
p = PGMImage('boy_noisy.pgm')

# TODO: Remove noise with frequency domain filtering

# Remove noise with Gaussian filtering
# (7 X 7)
gaussian_matrix_7 = Kernel(mask = [ [1,1,2,2,2,1,1], 
                                  [1,2,2,4,2,2,1], 
                                  [2,2,4,8,4,2,2], 
                                  [2,4,8,16,8,4,2], 
                                  [2,2,4,8,4,2,2], 
                                  [1,2,2,4,2,2,1], 
                                  [1,1,2,2,2,1,1] ])
# (15 X 15)
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

p_gaussian = spatially_filtered("boy_noisy.pgm", gaussian_matrix_7, normalize=True, truncate=False)
p_gaussian.save(f"boy_noisy_smoothed_gaussian-7") 

p_gaussian = spatially_filtered("boy_noisy.pgm", gaussian_matrix_15, normalize=True, truncate=False)
p_gaussian.save(f"boy_noisy_smoothed_gaussian-15")


# Experiment 2 (Convolution in Frequency Domain)

# Experiment 3 (Image Restoration - motion blur)

# Experiment 4 (Homomorphic filtering)