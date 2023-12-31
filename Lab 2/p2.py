#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import Image
from scipy.ndimage import filters
from scipy.signal import medfilt2d
import numpy as np
import matplotlib.pyplot as plt
import math as math
import glob
import os
from scipy import signal
import sys
import timeit

sys.path.append("../../p1/code") # set the path for visualPercepUtils.py
import visualPercepUtils as vpu


# -----------------------
# Salt & pepper noise
# -----------------------

def addSPNoise(im, percent):
    # Now, im is a PIL image (not a NumPy array)
    # percent is in range 0-100 (%)

    # convert image it to numpy 2D array and flatten it
    im_np = np.array(im)
    im_shape = im_np.shape  # keep shape for later use (*)
    im_vec = im_np.flatten()  # this is a 1D array # https://www.geeksforgeeks.org/differences-flatten-ravel-numpy/

    # generate random locations
    N = im_vec.shape[0]  # number of pixels
    m = int(math.floor(percent * N / 100.0)) # number of pixels corresponding to the given percentage
    locs = np.random.randint(0, N, m)  # generate m random positions in the 1D array (index 0 to N-1)

    # generate m random S/P values (salt and pepper in the same proportion)
    s_or_p = np.random.randint(0, 2, m)  # 2 random values (0=salt and 1=pepper)

    # set the S/P values in the random locations
    im_vec[locs] = 255 * s_or_p  # values after the multiplication will be either 0 or 255

    # turn the 1D array into the original 2D image
    im2 = im_vec.reshape(im_shape) # (*) here is where we use the shape that we saved earlier

    # convert Numpy array im2 back to a PIL Image and return it
    return Image.fromarray(im2)


def testSandPNoise(im, percents):
    imgs = []
    for percent in percents:
        imgs.append(addSPNoise(im, percent))
    return imgs


# -----------------
# Gaussian noise
# -----------------

def addGaussianNoise(im, sd=5):
    return np.clip(im + np.random.normal(0, sd, im.shape), 0, 255).astype(np.uint8)

def quotientImage(im, sigma):
    # Apply Gaussian blur to the original image
    blurred_im = gaussianFilter(im, sigma)

    # Compute the quotient image
    quotient_im = im / (blurred_im + 1e-8)  # Adding a small value to avoid division by zero

    return quotient_im

# -------------------------
# Average (or mean) filter
# -------------------------

def averageFilter(im, filterSize):
    mask = np.ones((filterSize, filterSize))
    mask = np.divide(mask, filterSize * filterSize)
    return filters.convolve(im, mask)

def averageFilterSep(im, filterSize):
    mask_row = np.ones((1, filterSize)) / filterSize
    mask_column = np.ones((filterSize, 1)) / filterSize

    convolver_rows = signal.convolve2d(im, mask_row, mode='same', boundary='wrap')
    convolved_image = signal.convolve2d(convolver_rows, mask_column, mode='same', boundary='wrap')
    return convolved_image
    

def testAverageFilter(im_clean, params):
    imgs = []
    for sp_pctg in params['sp_pctg']:
        im_dirty = addSPNoise(im_clean, sp_pctg) # salt and pepper noise
        for filterSize in params['filterSizes']:
            imgs.append(np.array(im_dirty))
            imgs.append(averageFilter(im_dirty, filterSize))
    return imgs


# -----------------
# Gaussian filter
# -----------------

def gaussianFilter(im, sigma=5):
    # im is PIL image
    return filters.gaussian_filter(im, sigma)

def explicitGaussianFilter(im, n=0, sigma=5):
    gv1d = signal.gaussian(n, std=sigma)
    gv2d = np.outer(gv1d, gv1d)
    gv2d /= np.sum(gv2d)

    # Convolve the image with the Gaussian kernel
    filtered_image = signal.convolve2d(im, gv2d, mode='same', boundary='wrap')

    return filtered_image

def explicitGaussianFilterSep(im, n=0, sigma=5):
    gv1d = signal.gaussian(n, std=sigma)
    
    gv1d = gv1d.reshape((1, n))

    filtered_rows = signal.convolve2d(im, gv1d, mode='same', boundary='wrap')

    filtered_image = signal.convolve2d(filtered_rows, gv1d.T, mode='same', boundary='wrap')
    
    return filtered_image

def testGaussianFilter(im_clean, params):
    # This function turned out to be too similar to testAverageFilter
    # This is a good sign that code factorization is called for :)
    imgs = []
    for sigma in params['sd_gauss_noise']:
        im_dirty = addGaussianNoise(im_clean, sigma)
        for filterSize in params['sd_gauss_filter']:
            imgs.append(np.array(im_dirty))
            imgs.append(gaussianFilter(im_dirty, filterSize))
    return imgs

def testGaussianNoise(im, sigmas):
    imgs = []
    for sigma in sigmas:
        imgs.append(im)
        imgs.append(addGaussianNoise(im.copy(), sigma))
    return imgs
# -----------------
# Median filter
# -----------------

def medianFilter(im, filterSize):
    return medfilt2d(im, filterSize)

def testMedianFilter(im_clean, params):
    # This function turned out to be too similar to testAverageFilter
    # This is a good sign that code factorization is called for :)
    imgs = []
    for sp_pctg in params['sp_pctg']:
        im_dirty = addSPNoise(im_clean, sp_pctg)
        for filterSize in params['filterSizes']:
            imgs.append(np.array(im_dirty))
            imgs.append(medianFilter(im_dirty, filterSize))
    return imgs

def testGaussianFilterForSPNoise(im, sigmas, percent):
    imgs = []
    for sigma in sigmas:
        im_dirty = addSPNoise(im, percent)
        im_filtered = gaussianFilter(im_dirty, sigma)
        imgs.append(np.array(im_dirty))
        imgs.append(im_filtered)
    return imgs

def testAverageFilterForGaussianNoise(im, filterSizes, sigma):
    imgs = []
    for filterSize in filterSizes:
        im_dirty = addGaussianNoise(im, sigma)
        im_filtered = averageFilter(im_dirty, filterSize)
        imgs.append(np.array(im_dirty))
        imgs.append(im_filtered)
    return imgs

def testMedianFilterForGaussianNoise(im, filterSizes, sigma):
    imgs = []
    for filterSize in filterSizes:
        im_dirty = addGaussianNoise(im, sigma)
        im_filtered = medianFilter(im_dirty, filterSize)
        imgs.append(np.array(im_dirty))
        imgs.append(im_filtered)
    return imgs

def testAverageFilterSeparableTimes(im, sizes):
    timesAverageFilter = []
    timesAverageFilterSep = []
    for size in sizes:
        timesAverageFilter.append(timeit.timeit(lambda: averageFilter(im, size), number=1))
        timesAverageFilterSep.append(timeit.timeit(lambda: averageFilterSep(im, size), number=1))

    plt.plot(sizes, timesAverageFilter, label='Average Filter')
    plt.plot(sizes, timesAverageFilterSep, label='Average Filter Separable')
    # Añadir etiquetas
    plt.xlabel('Tamaño del Filtro')
    plt.ylabel('Tiempo de Ejecución (segundos)')
    plt.title('Comparación de Tiempos de Ejecución')
    plt.legend()
    plt.show()

def testQuotientImage(im, sigmas):
    imgs = []
    for sigma in sigmas:
        imgs.append(im)
        imgs.append(quotientImage(im.copy(), sigma))
    return imgs

def testGaussianFilterSeparableTimes(im, sigmas):
    timesGaussianFilter = []
    timesGaussianFilterSep = []
    for sigma in sigmas:
        timesGaussianFilter.append(timeit.timeit(lambda: gaussianFilter(im, sigma), number=1))
        timesGaussianFilterSep.append(timeit.timeit(lambda: explicitGaussianFilterSep(im, sigma=sigma), number=1))

    plt.plot(sigmas, timesGaussianFilter, label='Gaussian Filter')
    plt.plot(sigmas, timesGaussianFilterSep, label='Gaussian Filter Separable')
    # Añadir etiquetas
    plt.xlabel('Tamaño del Filtro')
    plt.ylabel('Tiempo de Ejecución (segundos)')
    plt.title('Comparación de Tiempos de Ejecución')
    plt.legend()
    plt.show()

def testBothGaussiansFilter(im, sigma, n):
    first_gaussian = gaussianFilter(im, sigma)
    second_gaussian = explicitGaussianFilter(im, n, sigma)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].imshow(first_gaussian)
    axs[0].set_title('1D Gaussian Vector')
    axs[0].set_xlabel('Index')
    axs[0].set_ylabel('Value')


    axs[1].imshow(second_gaussian)
    axs[1].set_title('2D Gaussian Matrix')
    axs[1].set_xlabel('Column Index')
    axs[1].set_ylabel('Row Index')
    axs[1].axis('off')  # Hide axes for the image

    plt.tight_layout()
    plt.show()
# -----------------
# Test image files
# -----------------

path_input = './imgs-P2/'
path_output = './imgs-out-P2/'
bAllFiles = True
if bAllFiles:
    files = glob.glob(path_input + "*.ppm")
else:
    files = [path_input + 'lena256.pgm']  # lena256, lena512

# --------------------
# Tests to perform
# --------------------

testsNoises = ['testSandPNoise', 'testGaussianNoise']
testsFilters = ['testAverageFilter', 'testGaussianFilter', 'testMedianFilter']
bAllTests = True
if bAllTests:
    tests = testsNoises + testsFilters
else:
    tests = ['testSandPNoise']

# -------------------------------------------------------------------
# Dictionary of user-friendly names for each function ("test") name
# -------------------------------------------------------------------

nameTests = {'testGaussianNoise': 'Gaussian noise',
             'testSandPNoise': 'Salt & Pepper noise',
             'testAverageFilter': 'Mean filter',
             'testGaussianFilter': 'Gaussian filter',
             'testMedianFilter': 'Median filter'}

bSaveResultImgs = False

# -----------------------
# Parameters of noises
# -----------------------
percentagesSandP = [3]  # ratio (%) of image pixes affected by salt and pepper noise
gauss_sigmas_noise = [3, 5, 10, 15, 50]  # standard deviation (for the [0,255] range) for Gaussian noise

# -----------------------
# Parameters of filters
# -----------------------

gauss_sigmas_filter = [1.2]  # standard deviation for Gaussian filter
avgFilter_sizes = [3, 7, 15]  # sizes of mean (average) filter
medianFilter_sizes = [3, 7, 15]  # sizes of median filter

testsUsingPIL = ['testSandPNoise']  # which test(s) uses PIL images as input (instead of NumPy 2D arrays)


# -----------------------------------------
# Apply defined tests and display results
# -----------------------------------------

def doTests():
    print("Testing on", files)
    for imfile in files:
        im_pil = Image.open(imfile)
        im = np.array(im_pil)  # from Image to array

        # for test in tests:

        #     if test == "testGaussianNoise":
        #         params = gauss_sigmas_noise
        #         subTitle = r", $\sigma$: " + str(params)
        #     elif test == "testSandPNoise":
        #         params = percentagesSandP
        #         subTitle = ", %: " + str(params)
        #     elif test == "testAverageFilter":
        #         params = {}
        #         params['filterSizes'] = avgFilter_sizes
        #         params['sp_pctg'] = percentagesSandP
        #         subTitle = ", " + str(params)
        #     elif test == "testMedianFilter":
        #         params = {}
        #         params['filterSizes'] = avgFilter_sizes
        #         params['sp_pctg'] = percentagesSandP
        #         subTitle = ", " + str(params)
        #     elif test == "testGaussianFilter":
        #         params = {}
        #         params['sd_gauss_noise'] = gauss_sigmas_noise
        #         params['sd_gauss_filter'] = gauss_sigmas_filter
        #         subTitle = r", $\sigma_n$ (noise): " + str(gauss_sigmas_noise) + ", $\sigma_f$ (filter): " + str(gauss_sigmas_filter)
        #     if test in testsUsingPIL:
        #         outs_pil = eval(test)(im_pil, params)
        #         outs_np = vpu.pil2np(outs_pil)
        #     else:
        #         # apply test to given image and given parameters
        #         outs_np = eval(test)(im, params)
        #         print("num images", len(outs_np))
        #     print(len(outs_np))
        #     # display original image, noisy images and filtered images
        #     vpu.showInGrid([im] + outs_np, title=nameTests[test] + subTitle)
        out = testGaussianNoise(im, gauss_sigmas_noise)
        vpu.showInGrid(out, title=['Quotient Image', f'sigmas_n (noise): {gauss_sigmas_noise}'], subtitles=[f'original' if _ % 2 == 0 else f'$\\sigma$={i}' for i in gauss_sigmas_noise for _ in range(2)])
    # outs_np = testGaussianFilterForSPNoise(im, [1, 2, 3], 3)
    # vpu.showInGrid([im] + outs_np, title=['Gaussian Filter & SP Noise', 'sigmas_n (filter): [1, 2, 3], sp_p(noise): 3)'])
    # outs_np = testAverageFilterForGaussianNoise(im, avgFilter_sizes, 10)
    # vpu.showInGrid([im] + outs_np, title=['Average Filter & Gaussian Noise', f'sigmas_n (noise): [10], avg_size(filter): [{avgFilter_sizes}])'])
    # outs_np = testMedianFilterForGaussianNoise(im, medianFilter_sizes, 10)
    # vpu.showInGrid([im] + outs_np, title=['Median Filter & Gaussian Noise', f'sigmas_n (noise): [10], median_size(filter): [{medianFilter_sizes}])'])
       


if __name__ == "__main__":
    doTests()
