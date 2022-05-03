import gc
from pathlib import Path
import os
import cv2
import sys
from cv2 import GC_BGD
import numpy as np
from gui import MyGUI
from PyQt5.QtWidgets import QApplication
from utils import noisy, preprocess, detect_edges, line_something, create_gamma_lut, contour_bounding, morph, get_avg_GRAY_color, get_lines_inside_bounding_box, save_result
from utils import contour_ransac


IMAGES_PATH = Path(__file__).parent / 'imgs'
IMAGE_FILES =  tuple(Path(f) for f in IMAGES_PATH.glob('*.jpg'))
APP = QApplication([])
GUI = MyGUI(IMAGE_FILES)
LAST_INDEX = 0
SELECTED_NOISE = "No noise"
NOISE_AMOUNT = 0
TEST_MODE = False
TEST = 0
SELECTED_METHOD = "My method"


def update_image():
    global LAST_INDEX, GUI, TEST, SELECTED_METHOD
    selected_image = str(IMAGE_FILES[LAST_INDEX])
    filename = IMAGE_FILES[LAST_INDEX].name
    img_BGR = cv2.imread(selected_image, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY) 

    
   
    match SELECTED_NOISE:
        case 'Additive noise':
            og_img = noisy('gauss', img, NOISE_AMOUNT)
        case 'Salt-pepper noise':
            og_img = noisy('s&p', img, NOISE_AMOUNT)
        case _:
            og_img = img

    # Noise cancel and white threshold
    tr_img = preprocess(og_img, filename=filename, testMode=TEST_MODE)

    match SELECTED_METHOD:
        case 'My method':
            mask = contour_bounding(tr_img, filename=filename, testMode=TEST_MODE)
            mask[mask == 255] = cv2.GC_FGD
            mask[mask == 125] = cv2.GC_BGD
            mask[mask == 0] = cv2.GC_BGD
        case '"Inspired method"':
            try:
                mask = contour_ransac(tr_img, filename=filename, testMode=TEST_MODE, testVal = TEST)
                mask[mask == 255] = cv2.GC_FGD
                mask[mask == 125] = cv2.GC_BGD
                mask[mask == 0] = cv2.GC_BGD
            except Exception:
                GUI.update_og_img(og_img)
                GUI.update_tr_img(img_BGR)
                return


    # 255 -> FGD
    # 125 -> PR_BGD
    # 0 -> BGD
    
    try:
        temp = cv2.cvtColor(og_img, cv2.COLOR_BGR2RGB)
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)
        
        cv2.grabCut(temp,mask,(0,0,0,0),bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)
        

        mask2 = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')
        gc_img = cv2.bitwise_and(temp, temp, mask=mask2)
        #cv2.imshow("gc_img", gc_img)
        gc_img[gc_img > 0] = 255
        gc_img = cv2.cvtColor(gc_img, cv2.COLOR_RGB2GRAY)
        
        img_BGR[gc_img == 255] = (0,255,0)
    except Exception:
        pass

    
    

    GUI.update_og_img(og_img)
    GUI.update_tr_img(img_BGR)

def img_select(i):
    # called when select image cb changes
    global LAST_INDEX
    LAST_INDEX = i
    update_image()


def noise_select(b):
    global SELECTED_NOISE
    if b.isChecked():
        SELECTED_NOISE = b.text()
        print(f"SELECTED NOISE: {SELECTED_NOISE}")
        update_image()
    

def amount_select2(textbox):
    global NOISE_AMOUNT
    NOISE_AMOUNT = float(textbox.text().replace(',', '.'))
    print(f"NOISE AMOUNT: {NOISE_AMOUNT}")
    update_image()

def amount_select(sl):
    global TEST
    TEST = sl.value()
    print(f"AMOUNT: {TEST}")
    update_image()

def test_mode_switch(button):
    global TEST_MODE
    TEST_MODE = button.isChecked()
    print(f"TEST MODE: {button.isChecked()}")
    update_image()

def select_method(b):
    global SELECTED_METHOD
    if b.isChecked():
        SELECTED_METHOD = b.text()
        print(f"SELECTED METHOD: {SELECTED_METHOD}")
        update_image()

if __name__ == "__main__":
    print(f" {len(IMAGE_FILES)} test images found.")
    update_image()
    GUI.get_main().show()
    GUI.add_img_cb_handler(img_select)
    GUI.add_ns_selected_handler(noise_select)
    GUI.add_test_changed_handler(test_mode_switch)
    GUI.add_ns_amount_handler(amount_select2)
    GUI.add_ns_slider_handler(amount_select)
    GUI.add_method_selected_handler(select_method)
sys.exit(APP.exec_())
    