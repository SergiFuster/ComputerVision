#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import Image
import numpy as np
import glob
import os
import visualPercepUtils as vpu
import matplotlib.pyplot as plt


def histeq(im, nbins=256):
    imhist, bins = np.histogram(im.flatten(), list(range(nbins)), density=False)
    cdf = imhist.cumsum()  # cumulative distribution function (CDF) = cumulative histogram
    factor = 255 / cdf[-1]  # cdf[-1] = last element of the cumulative sum = total number of pixels)
    im2 = np.interp(im.flatten(), bins[:-1], factor*cdf)
    return im2.reshape(im.shape), cdf

def testHistEq(im):
    im2, cdf = histeq(im)
    return [im2, cdf]

def darkenImg(im,p=2):
    return (im ** float(p)) / (255 ** (p - 1)).astype(int) # try without the float conversion and see what happens


def brightenImg(im, p=2):
    return np.power(255.0 ** (p - 1) * im, 1. / p).astype(int)  # notice this NumPy function is different to the scalar math.pow(a,b)


def invertImg(im):
    return 255 - im


def checkBoardImg(im, m, n):
    m_pixels = im.shape[0] // m
    n_pixels = im.shape[1] // n
    for fil in range(m):
        for col in range(n):
            if fil % 2 == col % 2:
                im[fil*m_pixels:(fil+1)*m_pixels, col*n_pixels:(col+1)*n_pixels] = \
                    (invertImg(im[fil*m_pixels:(fil+1)*m_pixels, col*n_pixels:(col+1)*n_pixels]))
    return im


def rMultiHist(im, n):
    hist, _ = np.histogram(im.flatten(), list(range(256)), density=False)
    if n == 1:
        return hist
    res = [hist]
    x_step = im.shape[0] // 2
    y_step = im.shape[1] // 2
    for i in range(2):
        for j in range(2):
            if(i == 1 and j == 1):
                res.append(rMultiHist(im[i*x_step:,j*y_step:], n - 1))
            elif(i == 1):
                res.append(rMultiHist(im[i*x_step:,j*y_step:(j+1)*y_step], n - 1))
            elif(j == 1):
                res.append(rMultiHist(im[i*x_step:(i+1)*x_step,j*y_step:], n - 1))
            else:
                res.append(rMultiHist(im[i*x_step:(i+1)*x_step,j*y_step:(j+1)*y_step], n - 1))
    return res


def muliHist(im, n):
    return rMultiHist(im, n)


def expTransf(alpha, n, l0, l1, bInc=True):
    l_values = np.linspace(l0, l1, n)

    a = (l1 - l0) / (np.exp(-alpha * l1**2) - np.exp(-alpha * l0**2))
    b = l0 - a * np.exp(-alpha * l0**2)
    
    T_values = a * np.exp(-alpha * l_values ** 2) + b
    return T_values if bInc else np.flip(T_values)


def transfImage(im, f):
    return im * f / 255


def testDarkenImg(im):
    im2 = darkenImg(im, p=2) 
    return [im2]


def testBrightenImg(im):
    p=2
    im2=brightenImg(im, p)
    return [im2]


def testCheckBoardImg(im):
    m, n = 8, 8
    im2 = checkBoardImg(im.copy(), m, n)
    return [im2]


path_input = './imgs-P1/'
path_output = './imgs-out-P1/'
bAllFiles = True
if bAllFiles:
    files = glob.glob(path_input + "*.ppm")
else:
    files = [path_input + 'iglesia.pgm']  # iglesia,huesos

bAllTests = False
if bAllTests:
    tests = ['testHistEq', 'testBrightenImg', 'testDarkenImg', 'testCheckBoardImg']
else:
    tests = ['testBrightenImg']  # ['testBrightenImg']

nameTests = {'testHistEq': "Histogram equalization",
             'testBrightenImg': 'Brighten image',
             'testDarkenImg': 'Darken image',
             'testCheckBoardImg': 'Checkerboard Image'}
suffixFiles = {'testHistEq': '_heq',
               'testBrightenImg': '_br',
               'testDarkenImg': '_dk',
               'testCheckBoardImg': '_cb'}

bSaveResultImgs = True


def testMultiHist():
    for file in files:
        im = np.array(Image.open(file).convert('L'))
        mHist = muliHist(im, 2)
        vpu.showInGrid([im] + mHist, title="Multi-histogram")


def saveImg(imfile, suffix, im2):
    dirname, basename = os.path.dirname(imfile), os.path.basename(imfile)
    fname, fext = os.path.splitext(basename)
    pil_im = Image.fromarray(im2.astype(np.uint8))  # from array to Image
    pil_im.save(path_output + '//' + fname + suffix + fext)


def doTests():
    print("Testing on", files)
    for imfile in files:
        im = np.array(Image.open(imfile))  # from Image to array
        for test in tests:
            out = eval(test)(im)
            im2 = out[0]
            vpu.showImgsPlusHists(im, im2, title=nameTests[test])
            if len(out) > 1:
                vpu.showPlusInfo(out[1], "cumulative histogram" if test=="testHistEq" else None)
            if bSaveResultImgs:
                saveImg(imfile, suffixFiles[test], im2)


def debug():
    # Example usage
    alpha = 0.01
    l0 = 0
    l1 = 255

    for file in files:
        im = np.array(Image.open(file).convert('L'))
        n = im.shape[1]
        # Apply increasing and decreasing transformations
        inc_transformation = expTransf(alpha, n, l0, l1, bInc=True)
        dec_transformation = expTransf(alpha, n, l0, l1, bInc=False)
        print(f"Increasing transformation: {inc_transformation}")
        print(f"Decreasing transformation: {dec_transformation}")
        im2 = transfImage(im, inc_transformation)
        im3 = transfImage(im, dec_transformation)
        # Show results
        vpu.showInGrid([im, im2, im3], title="Increasing and decreasing transformations")


if __name__ == "__main__":
    # debug()
    doTests()



