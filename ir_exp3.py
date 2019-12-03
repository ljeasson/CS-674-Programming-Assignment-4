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
    p1 = T / (pi * (u*a + v*b))
    p2 = sin(pi * (u*a + v*b))
    p3 = cos(u*a + v*b) + sin(u*a + v*b)

    return p1 * p2 * p3

# Experiment 3 (Image Restoration - motion blur)
F = PGMImage("images/lenna.pgm")
F_pixels = []
for row in F.pixels:
    F_pixels.append([i for i in row])

# Apply motion blur to lenna
a = b = 0.1
T = 1
G = PGMImage("images/lenna.pgm")

G.save(f"Motion-blur-{F.name}")

# Apply Gaussian noise to motion-blurred lenna
for i in (1, 10, 100):
    G_noisy = PGMImage("Motion-blur-lenna.pgm")

    G_pixels_noisy = []
    for row in G.pixels:
        G_pixels_noisy.append([i for i in row])

    for row in range(len(F_pixels)):
        for column in range(len(F_pixels[row])):
            G_pixels_noisy[row][column] = G_pixels_noisy[row][column] + box_muller(i,0)

    G_noisy.pixels = G_pixels_noisy
    G_noisy.truncate_intensity_values()

    G_noisy.save(f"Motion-blurred-Gaussian-noise-stdev{i}-{F.name}")

# TODO: Apply Inverse Filtering to each degraded image

# TODO: Apply Wiener Filtering to each degraded image