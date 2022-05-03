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
from collections import defaultdict


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

def save_result(img, filename, bool, n, method=""):
    if not bool:
        return

    path = RESULTS_FOLDER / Path(filename).stem / method
    if not path.exists():
        os.makedirs(path)
    cv2.imwrite(str(path / f'{n}.jpg'), img)

def preprocess(img, filename='', testMode = False):
    save_result(img, f'{filename}', testMode, 0)
    # median blur + morphing for noise
    tr_img = morph(cv2.medianBlur(img, 7), dil_iters=2, er_iters=1, size=3)
    x,y = img.shape
    
    
    save_result(tr_img, f'{filename}', testMode, 1)
    val, asd = cv2.threshold(tr_img, 50, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #print(f"VAL: {val}")
    #1. filter the white color
    lower = np.array([int(val) + 25])
    upper = np.array([255])
    tr_img = cv2.inRange(tr_img,lower,upper)
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
    try:
        maxcount = max(len(v) for v in db.values())
        asd = [k for k, v in db.items() if len(v) == maxcount]
    except ValueError:
        asd = [0]
    return asd
 
def line_something(img, filename='', testMode= False, _test = 10):
    height, width = img.shape
    thickness = int(width/150) if width > height else int(height/150)
    
    LINE_GROUP_RESOLUTION = 18 # 180 / LINE_GROUP_RES = VARIANCE IN CLASSIFYNG LINES BY THEIR ANGLES
    MAX_DIFF = 180 / LINE_GROUP_RESOLUTION
    
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
    asd = cv2.merge([img.copy(), img.copy(), img.copy()])

    # find contours 
    contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE )
    tr_img = asd.copy()
    tr_img[:][:][:] = 0 

    asd = cv2.drawContours(asd, contours, -1, (0,255,75), 2)
    save_result(asd, filename=filename, bool=testMode, n=4, method="my-method")

    all_lines_by_bounding = defaultdict(list)
    all_lines = []

    # First group line inside every bb
    for i in contours:
        # get the bounding box for the current contour
        bx,by,bw,bh = cv2.boundingRect(i)

        lines = get_lines_inside_bounding_box(bx,by,bw,bh, img) # get the lines inside bb by THE MOST COMMON ANGLE
        if lines is not None: # if bb doesnt contain any lines, then ignore
            groups = group_lines_by_angle(lines)
            lines = groups[most_entries(groups)[0]]
            if True:
                cv2.rectangle(tr_img, (bx, by), (bx+bw, by+bh), (255,0,0), 3 )
                cv2.rectangle(asd, (bx, by), (bx+bw, by+bh), (255,0,0), 3 )
                for line in lines:
                    line[0][0] += bx
                    line[0][1] += by
                    line[0][2] += bx
                    line[0][3] += by
                    all_lines.append(line)
                    all_lines_by_bounding[(bx, by, bw, bh)].append(line)
                    l = line[0]
                    cv2.line(tr_img, (l[0], l[1]), (l[2], l[3]), (255,0,255),2, cv2.LINE_AA)

    save_result(asd, filename=filename, bool=testMode, n=5, method="my-method")
    save_result(tr_img, filename=filename, bool=testMode, n =6, method="my-method")
    
    # Then group every line on the img
    all_groups = group_lines_by_angle(all_lines, LINE_GROUP_RESOLUTION=4)
    # Basically group all the lines into four groups: 
    # every group[key] represents a list of lines whose angle to the x axis is between key*45 - (key*45) + 45
    # So if i divide into four, there will be for groups:
    # 0-45, 45-90, 90-135, 135-180
    # I basically want to separate "kinda" horizontal, and "kinda" vertical lines so 0-45 and 135-180 can be grouped together, same with the other

    for line in all_groups[3]:
        all_groups[0].append(line) # all_groups[0] are the "kinda" horizontal lines
    for line in all_groups[2]:
        all_groups[1].append(line) # all_groups[1] are the "kinda" vertical lines
    del all_groups[3]
    del all_groups[2]

    most_common_key = most_entries(all_groups)[0]
    test = asd.copy()
    test[:][:][:] = 0
    lines = all_groups[most_common_key]

    #for line in lines:
        #for key, value in all_lines_by_bounding.items():
            #bx, by, bw, bh = key
            #if line in value:
                #l = line[0]
                #cv2.rectangle(test, (bx, by), (bx+bw, by+bh), (255,255,255), -1 )
                #cv2.line(test, (l[0] + bx, l[1]+by), (l[2]+bx, l[3]+by), (255,0,255),2, cv2.LINE_AA)
    bbs_sofar = []
    for line in lines:
        l = line[0]
        for key, value in all_lines_by_bounding.items():
            bx, by, bw, bh = key
            l2 = value[0][0] #??

            if l2[0] == l[0] and l2[1] == l[1] and l2[2] == l[2] and l2[3] == l[3]:
                #cv2.line(test, (l[0], l[1]), (l[2], l[3]), (255,0,255),2, cv2.LINE_AA)
                cv2.rectangle(test, (bx, by), (bx+bw, by+bh), (255,255,255), -1 )
                bbs_sofar.append(key)

    #cv2.imshow("TEST", test)
    save_result(test, filename=filename, bool=testMode, n=7, method="my-method")
    # 255 -> FGD
    # 125 -> PR_BGD
    # 0 -> BGD
    test = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
    mask = test & img
    temp_mask = mask ^ test
    temp_mask[temp_mask == 255] = 125
    mask = temp_mask + mask
    #cv2.imshow("mask", mask)
    save_result(mask, filename=filename, bool=testMode, n = 8, method="my-method")
    return mask





def get_lines_inside_bounding_box(bx, by, bw, bh, img, maxLineGap = 3, minLineLength=3, add = 15):
    w, h = img.shape
    AMOUNT = add
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
    maxGap = int(diag_len/maxLineGap) 
    minLength = int(diag_len/minLineLength) 
    lines = cv2.HoughLinesP(bound_img, 1, (np.pi / 360), 50,None,minLength,maxGap)
    if lines is None:
        return None
    return lines
    # ONLY RETURN THE LINES WHOSE ANGLES ARE THE MOST COMMON INSIDE THE BOUNDING BOX
    


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


def get_avg_GRAY_color(img):
    average_color_row = np.average(img, axis=0)
    return math.floor(np.average(average_color_row, axis=0))


#get a line from a point and unit vectors
def lineCalc(vx, vy, x0, y0):
    scale = 10
    x1 = x0+scale*vx
    y1 = y0+scale*vy
    m = (y1-y0)/(x1-x0)
    b = y1-m*x1
    return m,b

#the angle at the vanishing point
def angle(pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    inner_product = x1*x2 + y1*y2
    len1 = math.hypot(x1, y1)
    len2 = math.hypot(x2, y2)
    print(len1)
    print(len2)
    a=math.acos(inner_product/(len1*len2))
    return a*180/math.pi 

#vanishing point - cramer's rule
def lineIntersect(m1,b1, m2,b2) : 
    #a1*x+b1*y=c1
    #a2*x+b2*y=c2
    #convert to cramer's system
    a_1 = -m1 
    b_1 = 1
    c_1 = b1

    a_2 = -m2
    b_2 = 1
    c_2 = b2

    d = a_1*b_2 - a_2*b_1 #determinant
    dx = c_1*b_2 - c_2*b_1
    dy = a_1*c_2 - a_2*c_1

    intersectionX = dx/d
    intersectionY = dy/d
    return intersectionX,intersectionY

def contour_ransac(img, filename='', testMode=False, testVal = 0):
    w, h = img.shape
    img_diag_len = int(math.sqrt((w**2 + h**2)))
    MIN_BB_DIAG_LEN = 170 #img_diag_len/10
    RADIUS = 250 #img_diag_len/5

    img_cont = cv2.merge([img.copy(), img.copy(), img.copy()])
    img_ransac = img_cont.copy()
    # find contours 
    contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE )


    img_cont = cv2.drawContours(img_cont, contours, -1, (0,255,75), 2)
    save_result(img_cont, filename=filename, bool=testMode, n=4, method="inspired-method")

    x_left = []
    y_left = []
    x_right = []
    y_right = []
    points_left =  []
    points_right = []
    boundedLeft = []
    boundedRight = []
    
    for c in contours:
        bx,by,bw,bh = cv2.boundingRect(c)
        bb_diag_len = int(math.sqrt((bw**2 + bh**2)))
        ar = bw / bh
        if bb_diag_len >= MIN_BB_DIAG_LEN:
            x_left.append(bx)
            y_left.append(by)
            points_left.append([bx,by])
            x_right.append(bx+bw)
            y_right.append(by)
            points_right.append([bx+bw, by])
            cv2.circle(img_cont,(int(bx),int(by)),5,(0,250,250),2) #circles -> left line
            cv2.circle(img_cont,(int(bx+bw),int(by)),5,(250,250,0),2) #circles -> right line
            cv2.rectangle(img_cont, (bx, by), (bx+bw, by+bh),(0,0,255), 2 )

    
    
    #calculate median average for each line
    medianR = np.median(points_right, axis=0)
    medianL = np.median(points_left, axis=0)

    points_left = np.asarray(points_left)
    points_right = np.asarray(points_right)

    save_result(img_cont, filename=filename, bool=testMode, n=5, method="inspired-method")
    #cv2.imshow("asd", img_cont)

    boundL = []
    boundR = []
    img_cont = cv2.drawContours(cv2.merge([img.copy(), img.copy(), img.copy()]), contours, -1, (0,255,75), 2)
    #4. are the points bounded within the median circle?
    for i in points_left:
        if (((medianL[0] - i[0])**2 + (medianL[1] - i[1])**2) < RADIUS**2) == True:
            boundL.append(i)
            cv2.circle(img_cont,(int(i[0]),int(i[1])),5,(0,250,250),2) #circles -> left line
            
    boundL = np.asarray(boundL)

    for i in points_right:
        if (((medianR[0] - i[0])**2 + (medianR[1] - i[1])**2) < RADIUS**2) == True:
            boundR.append(i)
            cv2.circle(img_cont,(int(i[0]),int(i[1])),5,(250,250,0),2) #circles -> right line


    save_result(img_cont, filename=filename, bool=testMode, n=6, method="inspired-method")

    boundR = np.asarray(boundR)

    #select the points enclosed within the circle (from the last part)
    x_left = np.asarray(boundL[:,0])
    y_left =  np.asarray(boundL[:,1]) 
    x_right = np.asarray(boundR[:,0]) 
    y_right = np.asarray(boundR[:,1])
    #transpose x of the right and the left line
    bxLeftT = np.array([x_left]).transpose()
    bxRightT = np.array([x_right]).transpose()

    # ------------------- RANSAC --------------
    from sklearn import linear_model, datasets
    model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())

    # Run RANSAC on LEFT POINTS
    ransacX = model_ransac.fit(bxLeftT, y_left)
    inlier_maskL = model_ransac.inlier_mask_ #right mask

    # Run RANSAC on RIGHT POINTS
    ransacY = model_ransac.fit(bxRightT, y_right)
    inlier_maskR = model_ransac.inlier_mask_ #right mask

    seed_right = None
    seed_left = None
    #draw RANSAC selected circles
    for i, element in enumerate(boundR[inlier_maskR]):
       # print(i,element[0])
        cv2.circle(img_ransac,(element[0],element[1]),10,(0,250,0),2) #circles -> right line
        if seed_right is None:
            seed_right = (element[0], element[1])
        
    for i, element in enumerate(boundL[inlier_maskL]):
       # print(i,element[0])
        cv2.circle(img_ransac,(element[0],element[1]),10,(0,100,250),2) #circles -> right line
        if seed_left is None:
            seed_left = (element[0], element[1])

    #6. Calcuate the intersection point of the bounding lines
    #unit vector + a point on each line
    vx, vy, x0, y0 = cv2.fitLine(boundL[inlier_maskL],cv2.DIST_L2,0,0.01,0.01) 
    vx_R, vy_R, x0_R, y0_R = cv2.fitLine(boundR[inlier_maskR],cv2.DIST_L2,0,0.01,0.01)

    #get m*x+b
    m_L,b_L=lineCalc(vx, vy, x0, y0)
    m_R,b_R=lineCalc(vx_R, vy_R, x0_R, y0_R)

    #calculate intersention 
    intersectionX,intersectionY = lineIntersect(m_R,b_R,m_L,b_L)
    intersectionX = int(intersectionX)
    intersectionY = int(intersectionY)

    # Init mask
    mask = img.copy()
    mask[:] = 0
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    #cv2.circle(mask,(sample_selected_left[0],sample_selected_left[1]),10,(255,0,0),-1) 
    #cv2.circle(mask,(sample_selected_right[0],sample_selected_right[1]),10,(0,255,0),-1) 
    
    
    #7. draw the bounding lines and the intersection point
    m = RADIUS*10 
    if (intersectionY < (w * (img_diag_len))/2 ):
        cv2.circle(img_ransac,(int(intersectionX),int(intersectionY)),10,(0,0,255),15)
        cv2.line(img_ransac,(int(x0-m*vx), int(y0-m*vy)), (int(x0+m*vx), int(y0+m*vy)),(255,0,0),3)
        cv2.line(img_ransac,(int(x0_R-m*vx_R), int(y0_R-m*vy_R)), (int(x0_R+m*vx_R), int(y0_R+m*vy_R)),(255,0,0),3)
        cv2.line(mask,(int(x0-m*vx), int(y0-m*vy)), (int(x0+m*vx), int(y0+m*vy)),(125,125,125),20)
        cv2.line(mask,(int(x0_R-m*vx_R), int(y0_R-m*vy_R)), (int(x0_R+m*vx_R), int(y0_R+m*vy_R)),(125,125,125),20)
    save_result(img_ransac, filename=filename, bool=testMode, n=7, method="inspired-method")
    

    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    seed_main = get_midpoint(seed_left, seed_right)
    _seed_left = get_midpoint(seed_left, (0, seed_left[1]))
    _seed_right = get_midpoint(seed_right, (img.shape[1], seed_right[1]))

    
    temp = img.copy()
    temp[temp == 0] = 125
    cv2.floodFill(mask, None, seed_main, 255)
    mask = mask & temp

    #cv2.floodFill(mask, None, _seed_left, 55)
    #cv2.floodFill(mask, None, _seed_right, 55)
    #mask[mask == 10] = 125
    
    #cv2.imshow("mask", mask)
    save_result(mask, filename=filename, bool=testMode, n=8, method="inspired-method")
    return mask



def get_midpoint(p1, p2):
    return (int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2))


