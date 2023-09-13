#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import Image
import numpy as np
import glob
import os
import visualPercepUtils as vpu

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
    return (im ** float(p)) / (255 ** (p - 1)) # try without the float conversion and see what happens


def brightenImg(im, p=2):
    return np.power(255.0 ** (p - 1) * im, 1. / p)  # notice this NumPy function is different to the scalar math.pow(a,b)


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


def testDarkenImg(im):
    im2 = darkenImg(im, p=2)  # Is "p=2" different here than in the function definition? Can we remove "p=" here?
    return [im2]


def testBrightenImg(im):
    p=2
    im2=brightenImg(im,p)
    return [im2]


def testCheckBoardImg(im):
    m, n = 8, 8
    im2 = checkBoardImg(im, m, n)
    return [im2]


path_input = './imgs-P1/'
path_output = './imgs-out-P1/'
bAllFiles = True
if bAllFiles:
    files = glob.glob(path_input + "*.pgm")
else:
    files = [path_input + 'iglesia.pgm'] # iglesia,huesos

bAllTests = True
if bAllTests:
    tests = ['testHistEq', 'testBrightenImg', 'testDarkenImg', 'testCheckBoardImg']
else:
    tests = ['testHistEq']  # ['testBrightenImg']

nameTests = {'testHistEq': "Histogram equalization",
             'testBrightenImg': 'Brighten image',
             'testDarkenImg': 'Darken image',
             'testCheckBoardImg': 'Checkerboard Image'}
suffixFiles = {'testHistEq': '_heq',
               'testBrightenImg': '_br',
               'testDarkenImg': '_dk',
               'testCheckBoardImg': '_cb'}

bSaveResultImgs = True


def saveImg(imfile, suffix, im2):
    dirname, basename = os.path.dirname(imfile), os.path.basename(imfile)
    fname, fext = os.path.splitext(basename)
    pil_im = Image.fromarray(im2.astype(np.uint8))  # from array to Image
    pil_im.save(path_output + '//' + fname + suffix + fext)


def doTests():
    print("Testing on", files)
    for imfile in files:
        im = np.array(Image.open(imfile).convert('L'))  # from Image to array
        for test in tests:
            out = eval(test)(im)
            im2 = out[0]
            vpu.showImgsPlusHists(im, im2, title=nameTests[test])
            if len(out) > 1:
                vpu.showPlusInfo(out[1], "cumulative histogram" if test=="testHistEq" else None)
            if bSaveResultImgs:
                saveImg(imfile, suffixFiles[test], im2)


def debug():
    i = 0
    for imfile in files:
        im = np.array(Image.open(imfile).convert('L'))
        checkboard = checkBoardImg(im, 5, 2)
        vpu.showImgsPlusHists(im, checkboard, "INVERTED")
        i += 1


if __name__ == "__main__":
    # debug()
    doTests()



