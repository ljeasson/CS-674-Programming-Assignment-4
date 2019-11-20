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

# Experiment 2 (Convolution in Frequency Domain)