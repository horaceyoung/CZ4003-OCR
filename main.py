import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import pytesseract
from difflib import Se

# Calculate skew angle of an image
def getSkewAngle(cvImage):
    # Prep image, copy, convert to gray scale, blur, and threshold
    newImage = cvImage.copy()
    blur = cv2.GaussianBlur(newImage, (9, 9), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Apply dilate to merge text into meaningful lines/paragraphs.
    # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
    # But use smaller kernel on Y axis to separate between different blocks of text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=5)

    # Find all contours
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)

    # Find largest contour and surround in min area box
    largestContour = contours[0]
    minAreaRect = cv2.minAreaRect(largestContour)

    # Determine the angle. Convert it to the value that was originally used to obtain skewed image
    angle = minAreaRect[-1]
    if angle < -45:
        angle = 90 + angle
    return -1.0 * angle


img = cv2.imread('sample01.png', 0)

ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 0)
th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 0)
titles = ['Original Image', 'Global Thresholding (v = 127)', 'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]

# thresholding
for i in range(4):
    plt.subplot(2, 2, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.figure(frameon=False)
plt.imshow(th3, 'gray')
plt.xticks([]), plt.yticks([])
plt.savefig('adaptive_threshold.png')

print(pytesseract.image_to_string('adaptive_threshold.png'))

# denoising
th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 5.5)
plt.figure(frameon=False)
plt.imshow(th3, 'gray')
plt.xticks([]), plt.yticks([])
plt.savefig('denoise.png')
print(pytesseract.image_to_string('denoise.png'))

# deskewing
skewed_angle = getSkewAngle(th3)
(h, w) = th3.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, -skewed_angle, 1.0)
th4 = cv2.warpAffine(th3, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

plt.figure(frameon=False)
plt.imshow(th4, 'gray')
plt.xticks([]), plt.yticks([])
plt.savefig('deskewing.png')

print(pytesseract.image_to_string('deskewing.png'))

#erosion - not optimal
kernel = np.ones((3,3),np.uint8)
erosion = cv2.erode(img,kernel,iterations = 2)

plt.figure(frameon=False)
plt.imshow(erosion, 'gray')
plt.xticks([]), plt.yticks([])
plt.savefig('erosion.png')

print(pytesseract.image_to_string('erosion.png'))

# resize the image to 300 dpi
def set_image_dpi(path, filename, dpi=(300, 300)):
    img = Image.open(path)
    img.save(filename, dpi=(300, 300))

img = cv2.medianBlur(img, 3)
img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 1.5)

kernel = np.ones((3,3),np.uint8)
img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
# This section contains all helper functions

# Calculate skew angle of an image
def get_skew_angle(cvImage):
    newImage = cvImage.copy()
    thresh = cv2.threshold(newImage, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=5)

    # Find all contours
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)

    # Find largest contour and surround in min area box
    largestContour = contours[0]
    minAreaRect = cv2.minAreaRect(largestContour)

    # Determine the angle. Convert it to the value that was originally used to obtain skewed image

    angle = minAreaRect[-1]
    if angle < -45:
        angle = 90 + angle
    return -1.0 * angle

def rotate_image(cvImage, angle):
    newImage = cvImage.copy()
    (h, w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return newImage

angle = get_skew_angle(img)
img = rotate_image(img, -1.0 * angle)

kernel_size = [3, 5, 7, 9, 11]
constants = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for kernel in kernel_size:
    for constant in constants:
        temp = img.copy()
        temp = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, kernel, constant)
        print('Accuracy with kernel size ' + str(kernel) + ' constant ' + str(constant))
        compute_accuracy(temp)