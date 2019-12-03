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
F = PGMImage("images/lenna.pgm")
F_pixels = []
for row in F.pixels:
    F_pixels.append([i for i in row])


# TODO: Apply motion blur to lenna
G = PGMImage("images/lenna.pgm")
G_pixels = []
for row in F.pixels:
    G_pixels.append([i for i in row])

for row in range(len(G_pixels)):
    for column in range(len(G_pixels[row])):
        G_pixels[row][column] = F_pixels[row][column] * H(row, column, T=1, a=0.1, b=0.1)

G.pixels = G_pixels
G.truncate_intensity_values()

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
def inverse_filtering(image, r):
    inv_filtered = PGMImage(image)
    inv_filtered_pixels = []
    for row in inv_filtered.pixels:
        inv_filtered_pixels.append([i for i in row])

    inv_filtered.pixels = inv_filtered_pixels
    inv_filtered.truncate_intensity_values()

    inv_filtered.save(f"Inverse_filtered-{r}-{inv_filtered.name}")

for img in ("Motion-blurred-Gaussian-noise-stdev1-lenna.pgm", 
        "Motion-blurred-Gaussian-noise-stdev10-lenna.pgm",
        "Motion-blurred-Gaussian-noise-stdev100-lenna.pgm"):
    for r in (40, 70, 85):
        inverse_filtering(img, r)


# TODO: Apply Wiener Filtering to each degraded image
def wiener_filtering(image, k):
    wiener_filtered = PGMImage(image)
    wiener_filtered_pixels = []
    for row in wiener_filtered.pixels:
        wiener_filtered_pixels.append([i for i in row])

    wiener_filtered.pixels = wiener_filtered_pixels
    wiener_filtered.truncate_intensity_values()

    wiener_filtered.save(f"Wiener_filtered-{k}-{wiener_filtered.name}")

for img in ("Motion-blurred-Gaussian-noise-stdev1-lenna.pgm", 
        "Motion-blurred-Gaussian-noise-stdev10-lenna.pgm",
        "Motion-blurred-Gaussian-noise-stdev100-lenna.pgm"):
    for k in (1, 10, 20, 50, 100):
        wiener_filtering(img, k)
