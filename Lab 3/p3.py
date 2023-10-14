#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import Image
from scipy.ndimage import filters
import numpy.fft as fft
import numpy as np
import matplotlib.pyplot as plt
import math as math
import glob
import os
import sys
import timeit

sys.path.append("../Lab 1") # set the path for visualPercepUtils.py
import visualPercepUtils as vpu

# ----------------------
# Fourier Transform
# ----------------------

def FT(im):
    # https://numpy.org/doc/stable/reference/generated/numpy.fft.fft2.html
    return fft.fftshift(fft.fft2(im))  # perform also the shift to have lower frequencies at the center


def IFT(ft):
    return fft.ifft2(fft.ifftshift(ft))  # assumes ft is shifted and therefore reverses the shift before IFT


def testFT(im, params=None):
    ft = FT(im)
    #print(ft.shape)
    phase = np.angle(ft)
    magnitude = np.log(np.absolute(ft))
    bMagnitude = True
    if bMagnitude:
        im2 = np.absolute(IFT(ft))  # IFT consists of complex number. When applied to real-valued data the imaginary part should be zero, but not exactly for numerical precision issues
    else:
        im2 = np.real(IFT(ft)) # with just the module we can't appreciate the effect of a shift in the signal (e.g. if we use fftshift but not ifftshift, or viceversa)
        # Important: one case where np.real() is appropriate but np.absolute() is not is where the sign in the output is relevant
    return [magnitude, phase, im2]


# -----------------------
# Convolution theorem
# -----------------------

# the mask corresponding to the average (mean) filter
def avgFilter(filterSize):
    mask = np.ones((filterSize, filterSize))
    return mask/np.sum(mask)


# apply average filter in the spatial domain
def averageFilterSpace(im, filterSize):
    return filters.convolve(im, avgFilter(filterSize))


# apply average filter in the frequency domain
def averageFilterFrequency(im, filterSize):
    filterMask = avgFilter(filterSize)  # the usually small mask
    filterBig = np.zeros_like(im, dtype=float)  # as large as the image (dtype is important here!)

    # Now, place filter (the "small" filter mask) at the center of the "big" filter

    ## First, get sizes
    w, h = filterMask.shape
    w2, h2 = w / 2, h / 2  # half width and height of the "small" mask
    W, H = filterBig.shape
    W2, H2 = W / 2, H / 2  # half width and height of the "big" mask

    ## Then, paste the small mask at the center using the sizes computed before as an aid
    filterBig[int(W2 - w2):int(W2 + w2), int(H2 - h2):int(H2 + h2)] = filterMask

    # FFT of the big filter
    filterBig = fft.ifftshift(filterBig)  # shift origin at upper-left corner

    # Finally, IFT of the element-wise product of the FT's
    return np.absolute(IFT(FT(im) * FT(filterBig)))  # both '*' and multiply() perform elementwise product

def gaussianFilterFrequency(im, filterSize, sigma):
    mask = gaussian_mask(im.shape, filterSize, sigma)

    im_ft = FT(im)
    mask_ft = FT(mask)

    element_wise_product = im_ft * mask_ft

    im_filtered = np.absolute(IFT(element_wise_product))
    im_filtered = fft.ifftshift(im_filtered)

    return im_filtered, np.abs(mask_ft), np.log(np.abs(element_wise_product))

def gaussianFilterSpace(im, filterSize, sigma):
    return filters.gaussian_filter(im, sigma)

def gaussian_mask(im_shape, size, sigma):
    center = size // 2
    x, y = np.meshgrid(np.arange(size) - center, np.arange(size) - center)
    mask = np.exp(-(x**2 + y**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)
    matrix_zeros = np.zeros(im_shape)
    # Determinar las coordenadas para colocar la matriz gaussiana en el centro de la matriz de ceros
    start_row = (matrix_zeros.shape[0] - mask.shape[0]) // 2
    start_col = (matrix_zeros.shape[1] - mask.shape[1]) // 2

    end_row = start_row + mask.shape[0]
    end_col = start_col + mask.shape[1]

    matrix_zeros[start_row:end_row, start_col:end_col] = mask

    return matrix_zeros

def testConvTheo(im, params=None):
    filterSize = params['filterSize']

    # image filtered with a convolution in spatial domain
    imFiltSpace = averageFilterSpace(im, filterSize)

    # image filtered in frequency domain
    imFiltFreq = averageFilterFrequency(im, filterSize)

    # How much do they differ?
    # To quantify the difference, we use the Root Mean Square Measure (https://en.wikipedia.org/wiki/Root_mean_square)
    margin = 5  # exclude some outer pixels to reduce the influence of border effects
    rms = np.linalg.norm(imFiltSpace[margin:-margin, margin:-margin] - imFiltFreq[margin:-margin, margin:-margin], 2) / np.prod(im.shape)
    print("Images filtered in space and frequency differ in (RMS):", rms)

    return [imFiltSpace, imFiltFreq]


# -----------------------------------
# High-, low- and band-pass filters
# -----------------------------------

# generic band-pass filter (both, R and r, given) which includes the low-pass (r given, R not)
# and the high-pass (R given, r not) as particular cases
def bandPassFilter(shape, r=None, R=None):
    n, m = shape
    m2, n2 = np.floor(m / 2.0), np.floor(n / 2.0)
    [vx, vy] = np.meshgrid(np.arange(-m2, m2 + 1), np.arange(-n2, n2 + 1))
    distToCenter = np.sqrt(vx ** 2.0 + vy ** 2.0)
    if R is None:  # low-pass filter assumed
        assert r is not None, "at least one size for filter is expected"
        filter = distToCenter<r # same as np.less(distToCenter, r)
    elif r is None:  # high-pass filter assumed
        filter = distToCenter>R # same as np.greater(distToCenter, R)
    else:  # both, R and r given, then band-pass filter
        if r > R:
            r, R = R, r  # swap to ensure r < R (alternatively, warn the user, or throw an exception)
        filter = np.logical_and(distToCenter<R, distToCenter>r)
    filter = filter.astype('float')  # convert from boolean to float. Not strictly required

    bDisplay = True
    if bDisplay:
        plt.imshow(filter, cmap='gray')
        plt.show()
        plt.title("The filter in the frequency domain")
        # Image.fromarray((255*filter).astype(np.uint8)).save('filter.png')

    return filter


def testBandPassFilter(im, params=None):
    r, R = params['r'], params['R']
    filterFreq = bandPassFilter(im.shape, r, R)  # this filter is already in the frequency domain
    filterFreq = fft.ifftshift(filterFreq)  # shifting to have the origin as the FT(im) will be
    return [np.absolute(fft.ifft2(filterFreq * fft.fft2(im)))]  # the filtered image


# -----------------
# Test image files
# -----------------
path_input = './imgs-P3/'
path_output = './imgs-out-P3/'
bAllFiles = True
if bAllFiles:
    files = glob.glob(path_input + "*.pgm")
else:
    files = [path_input + 'einstein.jpg']  # lena255, habas, mimbre

# --------------------
# Tests to perform
# --------------------
bAllTests = True
if bAllTests:
    tests = ['testFT', 'testConvTheo', 'testBandPassFilter']
else:
    tests = ['testFT']
    tests = ['testConvTheo']
    tests = ['testBandPassFilter']

# -------------------------------------------------------------------
# Dictionary of user-friendly names for each function ("test") name
# -------------------------------------------------------------------

nameTests = {'testFT': '2D Fourier Transform',
             'testConvTheo': 'Convolution Theorem (tested on mean filter)',
             'testBandPassFilter': 'Frequency-based filters ("high/low/band-pass")'
             }

bSaveResultImgs = False

testsUsingPIL = []  # which test(s) uses PIL images as input (instead of NumPy 2D arrays)

def my_mask(n):
    # Create a 2D array of shape (n, n) filled with zeros
    arr = -np.ones((n, n), dtype=int)

    # Fill the lower triangle and diagonal with -1
    arr[np.tril_indices(n, k=-1)] = 1

    # Fill the diagonal with 0
    np.fill_diagonal(arr, 0)

    return arr

def my_filter(im, n):
    mask = my_mask(n)
    filterBig = np.zeros_like(im, dtype=float)  # as large as the image (dtype is important here!)

    w, h = mask.shape
    w2, h2 = w / 2, h / 2
    W, H = filterBig.shape
    W2, H2 = W / 2, H / 2

    filterBig[int(W2 - w2):int(W2 + w2), int(H2 - h2):int(H2 + h2)] = mask

    filterBig = fft.ifftshift(filterBig)  # shift origin at upper-left corner

    return np.absolute(IFT(FT(im) * FT(filterBig)))


# -----------------------------------------
# Apply defined tests and display results
# -----------------------------------------

def doTests():
    print("Testing on", files)
    for imfile in files:
        im_pil = Image.open(imfile).convert('L')
        im = np.array(im_pil)  # from Image to array

        for test in tests:
            if test is "testFT":
                params = {}
                subTitle = ": I, |F|, ang(F), IFT(F)"
            elif test is "testConvTheo":
                params = {}
                params['filterSize'] = 7
                subTitle = ": I, I*M, IFT(FT(I).FT(M))"
            else:
                params = {}
                r,R = 5,None # for low-pass filter
                # 5,30 for band-pass filter
                # None, 30 for high-pass filter
                params['r'], params['R'] = r,R
                # let's assume r and R are not both None simultaneously
                if r is None:
                    filter="high pass" + " (R=" + str(R) + ")"
                elif R is None:
                    filter="low pass" + " (r=" + str(r) + ")"
                else:
                    filter="band pass" + " (r=" + str(r) + ", R=" + str(R) + ")"
                subTitle = ", " + filter + " filter"

            if test in testsUsingPIL:
                outs_pil = eval(test)(im_pil, params)
                outs_np = vpu.pil2np(outs_pil)
            else:
                # apply test to given image and given parameters
                outs_np = eval(test)(im, params)
            print("# images", len(outs_np))
            print(len(outs_np))

            vpu.showInGrid([im] + outs_np, title=nameTests[test] + subTitle)

def exercise1():
    im_pil = Image.open(np.random.choice(files)).convert('L')
    im = np.array(im_pil)  # from Image to array
    ft = FT(im)
    magnitudes = np.log(np.abs(ft))
    phases = np.angle(ft)
    max_magnitude = np.max(magnitudes)
    min_magnitude = np.min(magnitudes)
    # (b) Display a boxplot of the magnitude values
    plt.figure()
    plt.boxplot(magnitudes.flatten())
    plt.title('Boxplot of Magnitude')
    plt.ylabel('Magnitude')
    plt.show()

    # (c) Plot a histogram of the phase values
    plt.figure()
    plt.hist(phases.flatten(), bins=50, range=[-np.pi, np.pi])
    plt.title('Histogram of Phase')
    plt.xlabel('Phase')
    plt.ylabel('Frequency')
    plt.show()

def exercise2():
    im_pil = Image.open(np.random.choice(files)).convert('L')
    im = np.array(im_pil)  # from Image to array

    filter_size = 100
    sigma = 100

    gaussian_filtered_frequency, mask_ft, element_wise_product = gaussianFilterFrequency(im, filter_size, sigma)
    gaussian_filtered_space = gaussianFilterSpace(im, filter_size, sigma)

    mean_filtered_frequency = averageFilterFrequency(im, filter_size)

    margin = 5  # exclude some outer pixels to reduce the influence of border effects
    rms = np.linalg.norm(gaussian_filtered_space[margin:-margin, margin:-margin] - gaussian_filtered_frequency[margin:-margin, margin:-margin], 2) / np.prod(im.shape)
    print("Images filtered in space and frequency differ in (RMS):", rms)

    vpu.showInGrid([im, gaussian_filtered_space, gaussian_filtered_frequency, mean_filtered_frequency], title='Filtros Gaussianos', subtitles=["Original", "Gaussian Filter Space", "Gaussian Filter Frequency", "Mean Filter Frequency"])
    vpu.showInGrid([mask_ft, element_wise_product], title='Mascara y Producto Element-Wise', subtitles=["Mask", "Element-Wise Product"])

def exercise3():
    im = np.random.random((500, 500))

    sizes =[5, 15, 25, 50]
    sigma = 5

    times_frequency = []
    times_space = []

    for size in sizes:
        times_frequency.append(timeit.timeit(lambda : gaussianFilterFrequency(im, size, sigma), number=1))
        times_space.append(timeit.timeit(lambda : gaussianFilterSpace(im, size, sigma), number=1))

    # Crear un gráfico para cada tamaño de filtro
    plt.figure(figsize=(10, 5))
    plt.title(f'Image Size = {im.shape} | Sigma = {sigma}')
    plt.xlabel('Size')
    plt.ylabel('Tiempo (ms)')
    plt.plot(sizes, times_space, label='Space Domain')
    plt.plot(sizes, times_frequency, label='Frequency Domain', linestyle='--')
    plt.legend()
    plt.grid()
    plt.show()

    times_frequency, times_space = [], []
    im_sizes = [(100, 100), (500, 500), (1000, 1000), (2000, 2000)]
    for size in im_sizes:
        im = np.random.random(size)
        times_frequency.append(timeit.timeit(lambda : gaussianFilterFrequency(im, 5, sigma), number=1))
        times_space.append(timeit.timeit(lambda : gaussianFilterSpace(im, 5, sigma), number=1))

    plt.figure(figsize=(10, 5))
    plt.title(f'Sigma = {sigma} | Size = 5')
    plt.xlabel('Image Size(n x n)')
    plt.ylabel('Tiempo (ms)')
    plt.plot(im_sizes, times_space, label='Space Domain')
    plt.plot(im_sizes, times_frequency, label='Frequency Domain', linestyle='--')
    plt.legend()
    plt.grid()
    plt.show()

def exercice4():
    im_pil = Image.open(np.random.choice(files)).convert('L')
    im = np.array(im_pil)  # from Image to array

    filter_size = 5

    filtered_image = my_filter(im, filter_size)

    vpu.showInGrid([im, filtered_image], title='Filtro de Máscara', subtitles=["Original", "Filtered"])
    
def exercice5():
    files = glob.glob(path_input + "*.gif")
    images = []
    for file in files:
        images.append(np.array(Image.open(file).convert('L')))
    
    fts = []
    for image in images:
        fts.append(FT(image))
    
    lamda = 0.5
    comb = lamda * fts[0] + (1 - lamda) * fts[1]

    magnitudes = [np.log(np.abs(comb))]
    phases = [np.angle(comb)]

    for ft in fts:
        magnitudes.append(np.log(np.abs(ft)))
        phases.append(np.angle(ft))
    
    vpu.showInGrid(magnitudes, title='Magnitudes de las Imágenes', subtitles=["Inverse of the combination", "FT stp1", "FT stp2"])
    vpu.showInGrid(phases, title='Fases de las Imágenes', subtitles=["Inverse of the combination", "FT stp1", "FT stp2"])


if __name__ == "__main__":
    #doTests()
    exercice5()
