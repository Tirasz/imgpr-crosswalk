import cv2
import numpy as np
import random
from pathlib import Path
import os
import math
from PyQt5 import QtGui
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

RESULTS_FOLDER = Path(__file__).parent / 'imgs/results'



def convert_cv_qt(cv_img, display_width, display_height):
    """Convert from an opencv (BGR) image to QPixmap"""
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
    p = convert_to_Qt_format.scaled(display_width, display_height, Qt.KeepAspectRatio)
    return QPixmap.fromImage(p)



def noisy(noise_typ,image, _amount=0.004):
    """
    Add noise to input image
    noise_typ: 'gauss' | 's&p'
    """
    match noise_typ:
        case 'gauss':
            noise = np.zeros(image.shape[:2], np.int16)
            cv2.randn(noise, 0.0, 50.0 * _amount)
            imnoise1 = cv2.add(image, noise, dtype=cv2.CV_8UC1)
            return imnoise1
        case 's&p':
            row,col = image.shape
            s_vs_p = 0.5
            number_of_pixels = (row*col)*_amount
            out = np.copy(image)

            for i in range(int(number_of_pixels/2)):
                y = random.randint(0, row-1)
                x = random.randint(0, col-1)
                out[y][x] = 255

            for i in range(int(number_of_pixels/2)):
                y = random.randint(0, row-1)
                x = random.randint(0, col-1)
                out[y][x] = 0
            
            return out
        


def morph(img, size = 7, shape = cv2.MORPH_RECT, dil_iters = 1, er_iters = 2):
    struct = cv2.getStructuringElement(shape, (size, size))
    dilated = cv2.dilate(img, struct, iterations=dil_iters)
    eroded = cv2.erode(dilated, struct, iterations=er_iters)
    return cv2.dilate(eroded, struct)

def save_result(img, filename, bool, n):
    path = RESULTS_FOLDER / Path(filename).stem
    
    if bool:
        if not path.exists():
            os.mkdir(path)
        cv2.imwrite(str(path / f'{n}.jpg'), img)

def preprocess(img, filename='', testMode = False):
    # median blur + morphing for noise
    tr_img = morph(cv2.medianBlur(img, 7), dil_iters=2, er_iters=1, size=3)

    # Multiply, to make brighter spots a little brighter
    tr_img = cv2.multiply(tr_img, 1.13)
    save_result(tr_img, f'{filename}', testMode, 1)

    # Global treshold and then hist eq 
    th_lower = 140
    th_upper = 255
    tr_img[tr_img > th_upper] = th_upper
    tr_img[tr_img < th_lower] = th_lower
    tr_img = cv2.normalize(tr_img, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    save_result(tr_img, f'{filename}', testMode, 2)
    # Another treshold, and then morphing (0 dilations, 2 erosions)
    val, tr_img = cv2.threshold(tr_img, 100, 255, cv2.THRESH_BINARY)
    tr_img = morph(tr_img, dil_iters=0, er_iters=2, size=3)
    save_result(tr_img, f'{filename}', testMode, 3)
    return tr_img


# Kép normalizálása és megjelenítése
def display_image(window, image):
    disp = cv2.normalize(image, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    cv2.imshow(window, disp)

def detect_edges(img, filename='', testMode = False):
    blurred = cv2.GaussianBlur(img, (5, 5), 2.0)
    edges = cv2.Canny(blurred, 180, 190, None, 5, True)
    save_result(edges, f'{filename}', testMode, 4)
    
    MAGN_THRESH_PERCENT = 0.2
    ksize = 3 # 3x3 Sobel

    Ix = cv2.Sobel(img, cv2.CV_32FC1, 1, 0, None, ksize)
    #display_image('Ix', Ix)

    Iy = cv2.Sobel(img, cv2.CV_32FC1, 0, 1, None, ksize)
    #display_image('Iy', Iy)

    Imagn = cv2.magnitude(Ix, Iy)
    #display_image('Gradient magnitude', Imagn)

    magn_th = np.amax(Imagn) * MAGN_THRESH_PERCENT
    #print('magn_th =', magn_th)
    _, ImagnTh = cv2.threshold(Imagn, magn_th, 1.0, cv2.THRESH_BINARY)
    #display_image('Thresholded gradient magnitude', ImagnTh)

    return edges


def line_something(img, filename='', testMode= False):
    tr_img = img.copy()
    cdstP = cv2.merge([img.copy(), img.copy(), img.copy()])
    #lines = cv2.HoughLines(tr_img, 1, np.pi / 180, 130, None, 0,0)
    lines =  cv2.HoughLinesP(tr_img, 1, (np.pi / 360), 50, None, 25, 20)
    tr_img = cv2.merge([img.copy(), img.copy(), img.copy()])
    if lines is None:
        return
    for i in range(0, len(lines)):
        l = lines[i][0]
        cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)

    save_result(cdstP, f'{filename}', testMode, 5)
    return cdstP
def normalize_value(value, _min, _max):
    
    return (value - _min) / (_max - _min)