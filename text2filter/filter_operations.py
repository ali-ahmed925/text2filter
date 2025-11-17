# filter_operations.py
import cv2
import numpy as np
import pywt  # for wavelet denoising

def apply_median(image, ksize):
    return cv2.medianBlur(image, ksize)

def apply_gaussian(image, ksize, sigmaX):
    return cv2.GaussianBlur(image, tuple(ksize), sigmaX)

def apply_boxblur(image, ksize):
    return cv2.blur(image, tuple(ksize))

def apply_bilateral(image, d, sigmaColor, sigmaSpace):
    return cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace)

def apply_sobel(image, ksize):
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)
    mag = cv2.magnitude(grad_x, grad_y)
    return cv2.convertScaleAbs(mag)

def apply_laplacian_sharpen(image, ksize):
    lap = cv2.Laplacian(image, cv2.CV_64F, ksize=ksize)
    return cv2.convertScaleAbs(image - lap)

def apply_highboost(image, alpha):
    blur = cv2.GaussianBlur(image, (3,3), 1)
    mask = cv2.subtract(image, blur)
    return cv2.addWeighted(image, 1 + alpha, mask, alpha, 0)

def apply_unsharp(image, amount, radius):
    blur = cv2.GaussianBlur(image, (radius*2+1, radius*2+1), 0)
    mask = cv2.subtract(image, blur)
    return cv2.addWeighted(image, 1.0, mask, amount, 0)


def apply_cartoon(image, bilateral_d, sigmaColor, sigmaSpace, edges_ksize):
    img_color = cv2.bilateralFilter(image, bilateral_d, sigmaColor, sigmaSpace)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_edge = cv2.Canny(img_gray, 100, 200)
    img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2BGR)
    img_edge = cv2.blur(img_edge, (edges_ksize, edges_ksize))
    cartoon = cv2.bitwise_and(img_color, img_edge)
    return cartoon

def apply_nlm(image, h, templateWindowSize, searchWindowSize):
    return cv2.fastNlMeansDenoisingColored(image, None, h, h, templateWindowSize, searchWindowSize)

def apply_wavelet_denoise(image, wavelet, level, threshold_scale):

    img_float = image.astype(np.float32)
    coeffs = pywt.wavedec2(img_float, wavelet=wavelet, level=level)
    coeffs_thresh = []
    for c in coeffs:
        if isinstance(c, tuple):
            cH, cV, cD = c
            thresh = threshold_scale * np.std(cD)
            cH = pywt.threshold(cH, thresh, mode='soft')
            cV = pywt.threshold(cV, thresh, mode='soft')
            cD = pywt.threshold(cD, thresh, mode='soft')
            coeffs_thresh.append((cH, cV, cD))
        else:
            thresh = threshold_scale * np.std(c)
            coeffs_thresh.append(pywt.threshold(c, thresh, mode='soft'))
    denoised = pywt.waverec2(coeffs_thresh, wavelet=wavelet)
    denoised = np.clip(denoised, 0, 255).astype(np.uint8)
    return denoised

def apply_gradient(image, ksize):
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)
    mag = cv2.magnitude(grad_x, grad_y)
    return cv2.convertScaleAbs(mag)

