from pathlib import Path
import os
import cv2
import sys
import numpy as np
from gui import MyGUI
from PyQt5.QtWidgets import QApplication
from utils import noisy, morph


os.chdir(Path(__file__).parent.resolve())
IMAGES_PATH = Path("./imgs")
IMAGE_FILES =  tuple(Path(f) for f in IMAGES_PATH.glob('*.jpg'))
APP = QApplication([])
GUI = MyGUI(IMAGE_FILES)
LAST_INDEX = 0
SELECTED_NOISE = "No noise"
NOISE_AMOUNT = 0

def update_image():
    global LAST_INDEX, GUI
    selected_image = str(IMAGE_FILES[LAST_INDEX])
    img = cv2.cvtColor(cv2.imread(selected_image, cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)

    match SELECTED_NOISE:
        case 'Additive noise':
            og_img = noisy('gauss', img, NOISE_AMOUNT)
        case 'Salt-pepper noise':
            og_img = noisy('s&p', img, NOISE_AMOUNT)
        case _:
            og_img = img
    # median blur + morphing for noise
    tr_img = morph(cv2.medianBlur(og_img, 7), dil_iters=2, er_iters=1, size=3)
    tr_img = cv2.multiply(tr_img, 1.2)
    th_lower = 140
    th_upper = 255
    tr_img[tr_img > th_upper] = th_upper
    tr_img[tr_img < th_lower] = th_lower
    tr_img = cv2.normalize(tr_img, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    val, tr_img = cv2.threshold(tr_img, 100, 255, cv2.THRESH_BINARY)
    tr_img = morph(tr_img, dil_iters=0, er_iters=2, size=3)
    GUI.update_og_img(og_img)
    GUI.update_tr_img(tr_img)

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
    

def amount_select(sl):
    global NOISE_AMOUNT
    NOISE_AMOUNT = sl.value() / 100
    print(f"NOISE AMOUNT: {NOISE_AMOUNT}")
    update_image()

def amount_select2(textbox):
    global NOISE_AMOUNT
    NOISE_AMOUNT = float(textbox.text().replace(',', '.'))
    print(f"NOISE AMOUNT: {NOISE_AMOUNT}")
    update_image()


if __name__ == "__main__":
    print(f" {len(IMAGE_FILES)} test images found.")
    update_image()
    GUI.get_main().show()
    GUI.add_img_cb_handler(img_select)
    GUI.add_ns_selected_handler(noise_select)
    GUI.add_ns_slider_handler(amount_select)
    GUI.add_ns_amount_handler(amount_select2)

sys.exit(APP.exec_())
    