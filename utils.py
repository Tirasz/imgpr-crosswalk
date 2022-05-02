from tkinter.filedialog import askdirectory
from tokenize import group
import cv2
from cv2 import GaussianBlur
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
    save_result(img, f'{filename}', testMode, 0)
    # median blur + morphing for noise
    tr_img = morph(cv2.medianBlur(img, 7), dil_iters=2, er_iters=1, size=3)
    x,y = img.shape
    
    save_result(tr_img, f'{filename}', testMode, 1)
    val, asd = cv2.threshold(tr_img, 50, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    print(f"VAL: {val}")
    #1. filter the white color
    lower = np.array([int(val) + 25])
    upper = np.array([255])
    tr_img = cv2.inRange(img,lower,upper)
    #tr_img = morph(cv2.medianBlur(tr_img, 7), dil_iters=2, er_iters=1, size=3)
    save_result(tr_img, f'{filename}', testMode, 2)
    # Multiply, to make brighter spots a little brighter
    #tr_img = cv2.multiply(tr_img, 1.13)
     #2. erode the frame
    erodeSize = int(y / 30)
    erodeStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (erodeSize,1))
    erode = cv2.erode(tr_img, erodeStructure, (-1, -1))

    # Another treshold, and then morphing (0 dilations, 2 erosions)
    tr_img = morph(tr_img, size=3, dil_iters=1, er_iters=3)
    #val, tr_img = cv2.threshold(tr_img, 50, 255, cv2.THRESH_BINARY)
    
    save_result(tr_img, f'{filename}', testMode, 3)
    return tr_img


# Kép normalizálása és megjelenítése
def display_image(window, image):
    disp = cv2.normalize(image, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    cv2.imshow(window, disp)

def detect_edges(img, filename='', testMode = False):
    morphed = morph(img)
    blurred = cv2.GaussianBlur(morphed, (5, 5), 4.0)
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

def most_entries(db):
    maxcount = max(len(v) for v in db.values())
    return [k for k, v in db.items() if len(v) == maxcount]

def line_something(img, filename='', testMode= False, _test = 10):
    height, width = img.shape
    thickness = int(width/150) if width > height else int(height/150)
    
    LINE_GROUP_RESOLUTION = 18 # 180 / LINE_GROUP_RES = VARIANCE IN CLASSIFYNG LINES BY THEIR ANGLES
    MAX_DIFF = 180 / LINE_GROUP_RESOLUTION
    from collections import defaultdict
    groups = defaultdict(list) #{0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}

    #tr_img = cv2.GaussianBlur(img.copy(),(5,5),1)
    tr_img = img.copy()
    #cv2.imshow("asd", tr_img)
    cdstP = img.copy()
    cdstP[:][:] = 0
    cdstP = cv2.cvtColor(cdstP, cv2.COLOR_GRAY2BGR)
    #lines = cv2.HoughLines(tr_img, 1, np.pi / 180, 130, None, 0,0)
    lines =  cv2.HoughLinesP(tr_img, 1, (np.pi / 360), 50,None,thickness*5,10)
    
    if lines is None:
        return
    for i in range(0, len(lines)):
        l = lines[i][0]

        angle = angle_from_x_axis((l[0], l[1]), (l[2], l[3]))
        group = int(math.floor(angle / MAX_DIFF)) if int(math.floor(angle / MAX_DIFF)) != LINE_GROUP_RESOLUTION -1 else 0
        groups[group].append(l)
        cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (255,0,255),2, cv2.LINE_AA)

        
    groups[LINE_GROUP_RESOLUTION-1] = groups[0].copy()

    mostKeys = most_entries(groups)[0]
    
    for l in groups[mostKeys]:
        cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (255,255,0),2, cv2.LINE_AA)
        
    
    
    #for l in groups[_test]:





    
    save_result(cdstP, f'{filename}', testMode, 5)
    
    return cdstP

def group_lines_by_angle(lines, LINE_GROUP_RESOLUTION = 18):
    MAX_DIFF = 180 / LINE_GROUP_RESOLUTION
    from collections import defaultdict
    groups = defaultdict(list)
    for line in lines:
        l = line[0]
        angle = angle_from_x_axis((l[0], l[1]), (l[2], l[3]))
        group = int(math.floor(angle / MAX_DIFF)) if int(math.floor(angle / MAX_DIFF)) != LINE_GROUP_RESOLUTION -1 else 0
        groups[group].append(line)
    # Last and first group (0 degrees and 180 degrees) are pretty much the same
    return groups


"""

    cdstP = morph(cdstP, size=3, er_iters=1)
    lines =  cv2.HoughLinesP(cdstP.copy(), 1, (np.pi / 360)*10, 50, None,150,0)
    tr_img = cdstP.copy()
    tr_img[:] = 0
    for i in range(0, len(lines)):
        l = lines[i][0]
        bot_left = bottom_left([(l[0], l[1]), (l[2], l[3])])
        cv2.line(tr_img, (l[0], l[1]), (l[2], l[3]), (255,255,255),1, cv2.LINE_AA)
        #cv2.circle(tr_img, bot_left, 4, (255,0,0), -1)
"""


    
def angle_from_x_axis(p1, p2):
    delta_x = p1[0] - p2[0]
    delta_y = p1[1] - p2[1]
    return (180 - (math.degrees(math.atan2(delta_y, delta_x))))%180

    
def contour_bounding(img, filename='', testMode = False):
    height, width = img.shape
    thickness = int(width/150) if width > height else int(height/150)
    tr_img = img.copy()
    x, y = tr_img.shape
    
    #erodeSize = int(y / 30)
    #erodeStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (erodeSize,1))
    #tr_img = cv2.erode(tr_img, erodeStructure, (-1, -1))

    #

    asd = cv2.merge([tr_img.copy(), tr_img.copy(), tr_img.copy()])

    # find contours 
    contours, hierarchy = cv2.findContours(tr_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE )
    tr_img = asd.copy()
    tr_img[:][:][:] = 0 
    bw_width = 170 
 
    asd = cv2.drawContours(asd, contours, -1, (0,255,75), 2)
    #save_result(asd, filename=filename, bool=testMode, n=4)
    mask = asd.copy()
    mask[:][:][:] = 0 

    all_lines = []

    for i in contours:
        bx,by,bw,bh = cv2.boundingRect(i)
        lines = get_lines_inside_bounding_box(bx,by,bw,bh, img) # THE MOST COMMON ANGLE
        if lines is not None:
            #TEST
            if True:
                cv2.rectangle(tr_img, (bx, by), (bx+bw, by+bh), (255,0,0), 3 )
                cv2.rectangle(asd, (bx, by), (bx+bw, by+bh), (255,0,0), 3 )
                cv2.rectangle(mask, (bx, by), (bx+bw, by+bh), (255,255,255), -1 )
                for line in lines:
                    all_lines.append(line)
                    l = line[0]
                    cv2.line(tr_img, (l[0] + bx, l[1]+by), (l[2]+bx, l[3]+by), (255,0,255),2, cv2.LINE_AA)

    save_result(asd, filename=filename, bool=testMode, n=5)
    #cv2.imshow("asd", asd)


    

    line_img = detect_edges(img.copy())
    
    lines =  cv2.HoughLinesP(line_img, 1, (np.pi / 360), 50,None,thickness*5,10)
    line_img[:][:] = 0
    line_img = cv2.cvtColor(line_img, cv2.COLOR_GRAY2BGR)
    if lines is None:
        return
    for i in range(0, len(lines)):
        l = lines[i][0]

        #angle = angle_from_x_axis((l[0], l[1]), (l[2], l[3]))
        #group = int(math.floor(angle / MAX_DIFF)) if int(math.floor(angle / MAX_DIFF)) != LINE_GROUP_RESOLUTION -1 else 0
        #groups[group].append(l)
        cv2.line(line_img, (l[0], l[1]), (l[2], l[3]), (0,0,255),2, cv2.LINE_AA)

    shit = img.copy()
    shit = cv2.addWeighted(line_img, 0.5, mask, 0.5, 0, shit)
    #cv2.imshow("asdasd",shit)

    return (tr_img, cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY))


def get_lines_inside_bounding_box(bx, by, bw, bh, img):
    w, h = img.shape
    AMOUNT = 15
    # increasing bounding box
    bx = (bx - AMOUNT) if (bx - AMOUNT) > 0 else bx   
    by = (by - AMOUNT) if (by - AMOUNT) > 0 else by
    bh = (bh + AMOUNT) if (bh + AMOUNT + by) < h else bh
    bw = (bw + AMOUNT) if (bw + AMOUNT + bx) < w else bw
    diag_len = int(math.sqrt((bh**2 + bw**2)))

    # making bounding box a separate img from original
    bound_img = detect_edges(img[by:by+bh, bx:bx+bw])
    #cv2.imshow("bound", bound_img)
    lines_img = bound_img.copy()
    lines_img[:] = 0
    lines_img = cv2.cvtColor(lines_img, cv2.COLOR_GRAY2BGR)
    # Getting lines
    lines = cv2.HoughLinesP(bound_img, 1, (np.pi / 360), 50,None,int(diag_len/3),int(diag_len/3))
    if lines is None:
        return None
    # ONLY RETURN THE LINES WHOSE ANGLES ARE THE MOST COMMON INSIDE THE BOUNDING BOX
    groups = group_lines_by_angle(lines)
    return groups[most_entries(groups)[0]]



def bottom_left(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    #rect = np.zeros((4, 2), dtype = "float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    #s = pts.sum(axis = 1)
    #rect[0] = pts[np.argmin(s)]
    #rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    #rect[1] = pts[np.argmin(diff)]
    #rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return pts[np.argmax(diff)]

def normalize_value(value, _min, _max):
    return (value - _min) / (_max - _min)




def create_gamma_lut(gamma):
    """Gamma paraméter értéknek megfelelő 256 elemű keresőtábla generálása.
    A hatványozás miatt először [0, 1] tartományra kell konvertálni, majd utána vissza [0, 255] közé.
    """
    lut = np.arange(0, 256, 1, np.float32)
    lut = lut / 255.0
    lut = lut ** gamma
    lut = np.uint8(lut * 255.0)

    return lut