import numpy as np
from math import cos, sin, pi, e, sqrt, atan2
import matplotlib
import matplotlib.pyplot as plt
from PGM import PGMImage as PGM

f = lambda x, y: ([2, 3, 3, 4] * y)[x][y]
N = 4

def F(u):
    return (1/N) * sum(
        f(x) * (e ** ( (-complex(1) * 2 * pi * u * x) / N ) ) 
                for x in range(N-1) 
    )

def mag(a_b: complex):
    return sqrt( (a_b.real ** 2) + (a_b.imag ** 2))

def phase(a_b: complex):
    return atan2(a_b.imag, a_b.real)

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

cfft = my_cfft

p = PGM('zigzag.pgm')
a = np.fft.fft2(p.pixels)
ashift = np.fft.fftshift(a)
magnitude_spectrum = 20 * np.log(np.abs(ashift))

# Set Plot properties
# plt.scatter(a.real, a.imag)
# plt.axis([-1_000_000, 1_000_000, -400_000, 400_000])
plt.imshow(20 * np.log(np.abs(np.fft.fftshift(a))), cmap = 'gray')

# Display plot
plt.show()


from PGM import PGMImage
p = PGMImage('lenna.pgm')
pxls = p.pixels

import copy
fft_out_c = copy.deepcopy(pxls)

fft_out_py = np.fft.fft2(pxls)

fft_out_c = [my_cfft(row) for row in fft_out_c]

for i in range(p.cols):
    col = [row[i] for row in fft_out_c]
    col = my_cfft(col)
    for j in range(p.rows):
        fft_out_c[j][i] = col[j]

def cfft2(arr2, inverse=False):
    fft_out = [my_cfft(row, inverse) for row in arr2]
    
    cols = len(arr2[0])
    for i in range(cols):
        col = [row[i] for row in fft_out]
        col = my_cfft(col, inverse)
        
        for j in range(len(arr2)):
            fft_out[j][i] = col[j]
    
    return fft_out

fft_out_c[0] == fft_out_py[0]


###################################
# Experiment 1.a
a = [2,3,4,4]
fft_a = my_cfft(a)
print(fft_a)

print(my_cfft(fft_a, inverse=True))

def plot_all(a, fft_a): 
    N = len(a)
    plt.plot(range(N), a, label="f(x)")
    plt.plot(range(N), [_.real for _ in fft_a], label="FFT - Real Part")
    plt.plot(range(N), [_.imag for _ in fft_a], label="FFT - Imaginary Part")
    plt.plot(range(N), [mag(_) for _ in fft_a], label="FFT - Magnitude")
    plt.legend()

plot_all(a, fft_a)

###################################
# Experiment 1.b
#u = 8
#N = 128

f = lambda x: cos(2 * pi * 8 * x / 128)
samples = [ f(x) for x in range(128) ]
plt.plot(range(128), samples, label="cos(2πux/N)")

samples_normed = [ f(x) * (-1 ** x) for x in range(128) ]

from matplotlib import pyplot as plt
plt.plot(range(128), samples_normed, label="cos(2πux/N) * (-1)^x")

fft_samples = cfft(samples_normed)

plot_all(samples_normed, fft_samples)

###################################
# Experiment 1.c
# Open Rect_128.dat
samples = open('./Rect_128.dat').read().splitlines()
samples = [float(s) for s in samples]
samples = [(-1 ** x) * samples[x] for x in range(len(samples))]

plot_all(samples, cfft(samples))


###################################
# Experiment 2
import PGM
from importlib import reload as r; r(PGM)

def square_image(sq_sz):
    canvas = np.zeros((512, 512))
    
    for i in range(sq_sz):
        for j in range(sq_sz):
            x = int( 256 - (sq_sz / 2) + i )
            y = int( 256 - (sq_sz / 2) + j )
            canvas[x][y] = 255
                
    p = PGM.PGMImage('lenna.pgm')
    p.pixels = canvas
    p.rows = p.cols = 512
    p.truncate()
    p.save(f'square-{sq_sz}.pgm')
    
    return p
    
    # from PIL import Image
    # return Image.open(f'square-{sq_sz}.pgm')

from matplotlib import pyplot as plt
plt.imshow(square_image(32).pixels, plt.cm.gray)

###################################
# Experiment 2.b, 2.c
def plot_figures(figures, nrows = 1, ncols=1):
    """Plot a dictionary of figures.
    
    Source: https://stackoverflow.com/questions/11159436/multiple-figures-in-a-single-window

    Parameters
    ----------
    figures : <title, figure> dictionary
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """

    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows)
    for ind,title in enumerate(figures):
        axeslist.ravel()[ind].imshow(figures[title], cmap=plt.gray())
        axeslist.ravel()[ind].set_title(title)
        axeslist.ravel()[ind].set_axis_off()

# sq64 = square_image(64)
fft64 = cfft2(sq64.pixels)
# sq128 = square_image(128)
fft128 = cfft2(sq128.pixels)

npl = np.log
npa = np.abs
shift = np.fft.fftshift

# TODO: For report, describe analytical solution for rect()
ones = np.ones((512,512))
plot_figures(
    {"32x32 square": sq64.pixels,                            "64x64 square": sq128.pixels, 
     "FFT of 32x32 square": npl(np.add(ones, npa(fft64))),   "FFT of 64x64 square": npl(np.add(ones, npa(fft128)))
    }
, 2, 2)


###################################
# Experiment 3
p = PGM.PGMImage('lenna.pgm')
a = cfft2(p.pixels)

# TODO: Implement fftshift ourselves
plt.imshow(20 * np.fft.fftshift(np.log(np.abs(a))), cmap = 'gray')

# Experiment 3.a
a = cfft2(p.pixels)

for i in range(len(a)):
    for j in range(len(a[i])):
        a[i][j] = complex(mag(a[i][j]), 0)

a = cfft2(a, inverse=True)

plt.imshow(np.log(np.abs(a)), cmap = 'gray')

# Experiment 3.b
a = cfft2(p.pixels)

# TODO: prove that to set the magnitude equal to one, 
# set the real part to cos(theta) 
# and the imaginary part to sin(theta) where theta=tan-1(imag/real)

for i in range(len(a)):
    for j in range(len(a[i])):
        theta = atan2(a[i][j].imag, a[i][j].real)
        a[i][j] = complex(real=cos(theta), imag=sin(theta))

a = np.array(cfft2(a, inverse=True)).astype(float)
        
p2 = PGM.PGMImage('lenna.pgm')
p2.pixels = a
p2.normalize()
plt.imshow(p2.pixels, plt.cm.gray)