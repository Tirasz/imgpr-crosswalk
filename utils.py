import cv2
import numpy as np
import random
from pathlib import Path
import os

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
