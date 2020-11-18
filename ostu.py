import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread("sample01.png", 0)

histo = cv.calcHist([img], [0], None, [256], [0, 256])
histo_norm = histo.ravel() / histo.max()
q = histo_norm.cumsum()

bins = np.arange(256)

fn_min = np.inf
thresh = -1

for i in range(1, 256):
    q1, q2 = q[i], q[255] - q[i]
    p1, p2 = np.hsplit(histo_norm, [i])
    b1, b2 = np.hsplit(bins, [i])

    mean_1, mean_2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
    var_1, var_2 = np.sum(((b1 - mean_1)**2)*p1)/q1, np.sum(((b2-mean_2)**2)*p2)/q2

    fn = q1*var_1 + q2*var_2
    if fn < fn_min:
        fn_min = fn
        thresh = i

ret, th = cv.threshold(img, thresh, 255, cv.THRESH_BINARY)

plt.figure()
plt.imshow(th, cmap='gray')
plt.savefig("ostu_out.png")
