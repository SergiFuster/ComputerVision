from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import filters
import glob
import sys

from skimage import feature
from skimage.transform import hough_line, hough_line_peaks  # , probabilistic_hough_line

from scipy import ndimage as ndi
from copy import deepcopy

sys.path.append("../../p1/code")
import visualPercepUtils as vpu

bLecturerVersion=False
# try:
#     import p4e
#     bLecturerVersion=True
# except ImportError:
#     pass # file only available to lecturers

def testSobel(im, params=None):
    gx = filters.sobel(im, 1)
    return [gx]

def testCanny(im, params=None):
    sigma = params['sigma']
    edge = feature.canny(im, sigma=sigma, low_threshold=0.2 * 255, high_threshold=0.25 * 255, use_quantiles=False)
    return [edge]


def testHough(im, params=None):
    edges = testCanny(im, params)[0]
    numThetas = 200
    H, thetas, rhos = hough_line(edges, np.linspace(-np.pi/2, np.pi/2, numThetas))
    print("# angles:", len(thetas))
    print("# distances:", len(rhos))
    print("rho[...]",rhos[:5],rhos[-5:])
    return [np.log(H+1), (H, thetas, rhos)] # log of Hough space for display purpose


def findPeaks(H, thetas, rhos, nPeaksMax=None):
    if nPeaksMax is None:
        nPeaksMax = np.inf
    return hough_line_peaks(H, thetas, rhos, num_peaks=nPeaksMax, threshold=0.15 * np.max(H), min_angle=20, min_distance=15)


# -----------------
# Test image files
# -----------------
path_input = './imgs-P4/'
path_output = './imgs-out-P4/'
bAllFiles = True
if bAllFiles:
    files = glob.glob(path_input + "*.p??")
else:
    files = [path_input + 'cuadros.png']  # cuadros, lena

# --------------------
# Tests to perform
# --------------------
bAllTests = True
if bAllTests:
    tests = ['testSobel', 'testCanny', 'testHough']
else:
    tests = ['testSobel']
    tests = ['testCanny']
    tests = ['testHough']

# -------------------------------------------------------------------
# Dictionary of user-friendly names for each function ("test") name
# -------------------------------------------------------------------

nameTests = {'testSobel': 'Detector de Sobel',
             'testCanny': 'Detector de Canny',
             'testHough': 'Transformada de Hough'}

bAddNoise = True
bRotate = False


def doTests():
    print("Testing on", files)
    nFiles = len(files)
    nFig = None
    for test in tests:
        if test is "testSobel":
            params = {}
        elif test in ["testCanny", "testHough"]:
            params = {}
            params['sigma'] = 5  # 15
        if test is "testHough":
            pass  # params={}

        for i, imfile in enumerate(files):
            print("testing", test, "on", imfile)

            im_pil = Image.open(imfile).convert('L')
            im = np.array(im_pil)  # from Image to array

            if bRotate:
                im = ndi.rotate(im, 15, mode='nearest')

            if bAddNoise:
                im = im + np.random.normal(loc=0, scale=5, size=im.shape)

            outs_np = eval(test)(im, params)
            print("num ouputs", len(outs_np))
            if test is "testHough":
                outs_np_plot = outs_np[0:1]
            else:
                outs_np_plot = outs_np
            nFig = vpu.showInFigs([im] + outs_np_plot, title=nameTests[test], nFig=nFig, bDisplay=True)  # bDisplay=True for displaying *now* and waiting for user to close

            if test is "testHough":
                H, thetas, rhos = outs_np[1]  # second output is not directly displayable
                peaks_values, peaks_thetas, peaks_rhos = findPeaks(H, thetas, rhos, nPeaksMax=None)
                vpu.displayHoughPeaks(H, peaks_values, peaks_thetas, peaks_rhos, thetas, rhos)
                if bLecturerVersion:
                    p4e.displayLines(im, peaks_thetas, peaks_rhos, peaks_values) # exercise
                    plt.show(block=True)
                # displayLineSegments(...) # optional exercise

    plt.show(block=True)  # show pending plots (useful if we used bDisplay=False in vpu.showInFigs())


if __name__ == "__main__":
    doTests()
