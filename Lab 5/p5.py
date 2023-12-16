# !/usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
import glob
import sys

from skimage import measure

from skimage.morphology import disk, square, closing, opening # for the mathematically morphology part

sys.path.append("../Lab 1/")
import visualPercepUtils as vpu

bStudentVersion=True
if not bStudentVersion:
    import p5e

def testOtsu(im, params=None):
    nbins = 256
    th = threshold_otsu(im)
    hist = np.histogram(im.flatten(), bins=nbins, range=[0, 255])[0]
    return [th, im > th, hist]  # threshold, binarized image (using such threshold), and image histogram


def fillGaps(im, params=None):
    binIm = im > threshold_otsu(im)
    sElem = disk(params['closing-radius'])
    return [binIm, closing(binIm, sElem)]

# Don't worry about this function
def removeSmallRegions(im, params=None):
    binIm = im > threshold_otsu(im)
    sElem = disk(params['opening-radius'])
    return [binIm, opening(binIm, sElem)]

# Don't worry about this function
def fillGapsThenRemoveSmallRegions(im, params=None):
    binIm, closeIm = fillGaps(im, params)  # first, fill gaps
    sElem = disk(params['opening-radius'])
    return [binIm, opening(closeIm, sElem)]

def labelConnectedComponents(im, params=None):
    binIm = im > threshold_otsu(im, params)
    binImProc = fillGapsThenRemoveSmallRegions(im, params)[1]
    return [binIm, binImProc,
            measure.label(binIm, background=0), measure.label(binImProc, background=0)]

def reportPropertiesRegions(labelIm,title):
    print("* * "+title)
    regions = measure.regionprops(labelIm)
    for r, region in enumerate(regions):  # enumerate() is often handy: it provides both the index and the element
        print("Region", r + 1, "(label", str(region.label) + ")")
        print("\t area: ", region.area)
        print("\t perimeter: ", round(region.perimeter, 1))  # show only one decimal place

# -----------------
# Test image files
# -----------------
path_input = './imgs-P5/'
path_output = './imgs-out-P5/'
bAllFiles = False
if bAllFiles:
    files = glob.glob(path_input + "*.p??")
else:
    files = [path_input + 'monedas.pgm']

# --------------------
# Tests to perform
# --------------------
bAllTests = True
if bAllTests:
    tests = ['testOtsu', 'labelConnectedComponents']
else:
    tests = ['fillGaps']
    tests = ['labelConnectedComponents']

# -------------------------------------------------------------------
# Dictionary of user-friendly names for each function ("test") name
# -------------------------------------------------------------------

nameTests = {'testOtsu': "thresholding with Otsu's method",
             'labelConnectedComponents': 'Labelling conected components'}

myThresh = 180  # use your own value here
diskSizeForClosing = 2  # don't worry about this
diskSizeForOpening = 5  # don't worry about this

def doTests():
    print("Testing ", tests, "on", files)
    nFiles = len(files)
    nFig = None
    for i, imfile in enumerate(files):
        im_pil = Image.open(imfile).convert('L')
        im = np.array(im_pil)  # from Image to array

        for test in tests:
            title = nameTests[test]
            print(imfile, test)
            if test is "testOtsu":
                params = {}
            elif test is "labelConnectedComponents":
                params = {}
                params['closing-radius'] = diskSizeForClosing
                params['opening-radius'] = diskSizeForOpening
                subtitles = ["original image", "binarized image", "Processed binary image", "Connected components", "Connected componentes from processed binary image"]

            outs_np = eval(test)(im, params)

            if test is "testOtsu":
                outs_np_plot = [outs_np[2]] + [outs_np[1]] + [im > myThresh]
                subtitles = ["original image", "Histogram", "Otsu with threshold=" + str(outs_np[0]),
                             "My threshold: " + str(myThresh)]
                m = n = 2
            else:
                outs_np_plot = outs_np
            print(len(outs_np_plot))
            vpu.showInGrid([im] + outs_np_plot, m=m, n=n, title=title, subtitles=subtitles)
            if test is 'labelConnectedComponents':
                plt.figure()
                labelImOriginalBinaryImage = outs_np_plot[2]
                labelImProcessedBinaryImage = outs_np_plot[3]
                vpu.showImWithColorMap(labelImOriginalBinaryImage,'jet') # the default color map, 'spectral', does not work in lab computers
                plt.show(block=True)
                titleForBinaryImg = "From unprocessed binary image"
                titleForProcesImg = "From filtered binary image"
                reportPropertiesRegions(labelIm=labelImOriginalBinaryImage,title=titleForBinaryImg)
                reportPropertiesRegions(labelIm=labelImProcessedBinaryImage,title=titleForProcesImg)

                if not bStudentVersion:
                    p5e.displayImageWithCoins(im,labelIm=labelImOriginalBinaryImage,title=titleForBinaryImg)
                    p5e.displayImageWithCoins(im,labelIm=labelImProcessedBinaryImage,title=titleForProcesImg)

    plt.show(block=True)  # show pending plots

def introduction():
    im = np.array(Image.open('./imgs-P5/monedas.pgm').convert('L'))
    plt.hist(im.flatten(), bins=256)
    plt.show()
    th = 100
    th_otsu = threshold_otsu(im)

    fig, axs = plt.subplots(1, 2, figsize=(10, 10))
    axs[0].imshow(im > th)
    axs[0].set_title(f'Threshold {th}')
    axs[1].imshow(im > th_otsu)
    axs[1].set_title(f'Otsu threshold {th_otsu}')
    plt.show()


    params = {}
    params['closing-radius'] = diskSizeForClosing
    params['opening-radius'] = diskSizeForOpening
    
    imgs = labelConnectedComponents(im, params)
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs[0][0].imshow(imgs[0])
    axs[0][0].set_title(f'Original image')
    axs[0][1].imshow(imgs[1])
    axs[0][1].set_title(f'Processed image')
    axs[1][0].imshow(imgs[2])
    axs[1][0].set_title(f'Connected components')
    axs[1][1].imshow(imgs[3])
    axs[1][1].set_title(f'Connected components from processed image')
    plt.show()

def fillGaps2(im, params):
    binIm = im > params['threshold']
    sElem = disk(params['closing-radius'])
    return closing(binIm, sElem)

def ejercicio1():
    im = np.array(Image.open('./imgs-P5/monedas2.pgm').convert('L'))
    my_threshold = 136

    filled = fillGaps2(im, {'closing-radius' : 2, 'threshold' : my_threshold})
    labeled = measure.label(filled, background=0)

    plt.imshow(labeled)
    plt.show()

    return labeled

def ejercicio2():
    labeled = ejercicio1()
    regions = measure.regionprops(labeled)

    circularity_threshold = 0.9
    coins_image = np.zeros(labeled.shape, dtype=np.uint8)
    coin_value = 1
    no_coin_value = 0
    for region in regions:
        area = region.area
        perimeter = region.perimeter
        circularity = (4 * np.pi * area) / (perimeter**2)
        print(f'Circularity of label {region.label}: {circularity}')

        if circularity > circularity_threshold:
            coins_image[labeled == region.label] = coin_value
        else:
            coins_image[labeled == region.label] = no_coin_value
    
    plt.imshow(coins_image, cmap='gray')
    plt.show()

    return coins_image

def ejercicio3():
    coins_image = ejercicio2()
    labeled = measure.label(coins_image, background=0)
    regions = measure.regionprops(labeled)
    result = np.zeros((coins_image.shape[0], coins_image.shape[1], 3), dtype=np.uint8)
    for region in regions:
        area = region.area
        print(f'Area of label {region.label}: {area}')
        diameter = np.sqrt(area / np.pi) * 2
        print(f'Diameter of label {region.label}: {diameter}')
        ratio = 2.54 / 50
        print(f'Ratio: {ratio}')
        diameter_m = diameter * ratio / 100
        print(f'Diameter in meters: {diameter_m}')
        if diameter_m > 0.02:
            result[labeled == region.label] = [255, 0, 0]
        else:
            result[labeled == region.label] = [0, 0, 255]

    plt.imshow(result)
    plt.show()

    return result

def ejercicio4(im=None):
    if im is None:
        coins_image = ejercicio2()
    else:
        coins_image = im
    labeled = measure.label(coins_image, background=0)
    regions = measure.regionprops(labeled)
    result = 0
    img = np.zeros((coins_image.shape[0], coins_image.shape[1], 3), dtype=np.uint8)
    for region in regions:
        area = region.area
        diameter = np.sqrt(area / np.pi) * 2
        ratio = 2.54 / 50
        diameter_m = diameter * ratio / 100
        if diameter_m > 0.02:
            result += 1
            img[labeled == region.label] = [255, 0, 0]
        else:
            result += 0.1
            img[labeled == region.label] = [0, 0, 255]

    plt.imshow(img)
    plt.title(f'Money = {result}€')
    plt.show()
    return result


if __name__ == "__main__":
    # doTests()
    pass