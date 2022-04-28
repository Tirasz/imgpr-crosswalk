from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget, QWidget, QRadioButton, QComboBox, QSlider, QLabel, QLineEdit
from PyQt5.QtGui import QIntValidator,QDoubleValidator,QFont
from utils import convert_cv_qt

class MyGUI:
    def select_noise_amount(self,sl):
        pass
        #print(f"SELECETED NOISE AMOUNT: {sl.value()}")

    def select_image(self, i):
        pass
        #print(f"SELECTED IMAGE: {self.IMAGE_FILES[i]}")

    def select_noise(self, b):
        pass
        #if b.isChecked():
            #print(f"SELECTED NOISE: {b.text()}")

    def __init__(self, IMAGE_FILES = []):
        self.IMAGE_FILES = IMAGE_FILES

        self.window = QWidget()
        self.window.setWindowTitle('Crosswalk detection')
        #MAX_WIDTH = app.primaryScreen().size().width()
        #MAX_WIDTH = int(MAX_WIDTH - (MAX_WIDTH/7))
        #self.window.setFixedWidth(MAX_WIDTH)

        # THE LAYOUTS
        self.main_layout = QVBoxLayout()
        self.options_layout = QHBoxLayout()
        self.images_layout = QHBoxLayout()

        # SELECT IMAGE COMBO BOX
        self.selected_image_cb = QComboBox()
        self.selected_image_cb.addItems([str(x) for x in self.IMAGE_FILES])
        self.selected_image_cb.currentIndexChanged.connect(self.select_image)
        self.options_layout.addWidget(self.selected_image_cb)

        # SELECT NOISE RADIO BUTTONS
        self.no_noise_rb = QRadioButton("No noise")
        self.no_noise_rb.setChecked(True)
        self.no_noise_rb.toggled.connect(lambda:self.select_noise(self.no_noise_rb))
        self.ad_noise_rb = QRadioButton("Additive noise")
        self.ad_noise_rb.toggled.connect(lambda:self.select_noise(self.ad_noise_rb))
        self.sp_noise_rb = QRadioButton("Salt-pepper noise")
        self.sp_noise_rb.toggled.connect(lambda:self.select_noise(self.sp_noise_rb))
        self.options_layout.addWidget(self.no_noise_rb)
        self.options_layout.addWidget(self.ad_noise_rb)
        self.options_layout.addWidget(self.sp_noise_rb)

        # Noise slider
        self.noise_slider = QSlider(Qt.Horizontal)
        self.noise_slider.setMinimum(0)
        self.noise_slider.setMaximum(100)
        self.noise_slider.setValue(0)
        self.noise_slider.setTickPosition(QSlider.TicksBelow)
        self.noise_slider.setTickInterval(5)
        self.noise_slider.valueChanged.connect(lambda:self.select_noise_amount(self.noise_slider))
        #self.options_layout.addWidget(self.noise_slider)

        # Noise text input
        self.noise_amount = QLineEdit()
        self.noise_amount.setValidator(QDoubleValidator(0,1,2))
        self.options_layout.addWidget(self.noise_amount)

        # Image labels
        self.og_img_label = QLabel()
        self.tr_img_label = QLabel()
        self.images_layout.addWidget(self.og_img_label)
        self.images_layout.addWidget(self.tr_img_label)


        self.main_layout.addLayout(self.options_layout)
        self.main_layout.addLayout(self.images_layout)

        self.window.setLayout(self.main_layout)

    def get_main(self):
        return self.window

    def add_ns_amount_handler(self, handler):
        "handler(text)"
        self.noise_amount.editingFinished.connect(lambda:handler(self.noise_amount))

    def add_ns_slider_handler(self, handler):
        "handler(self.noise_slider)"
        self.noise_slider.valueChanged.connect(lambda:handler(self.noise_slider))
    
    def add_ns_selected_handler(self, handler):
        "handler(self.**_noise_rb)"
        self.no_noise_rb.toggled.connect(lambda:handler(self.no_noise_rb))
        self.ad_noise_rb.toggled.connect(lambda:handler(self.ad_noise_rb))
        self.sp_noise_rb.toggled.connect(lambda:handler(self.sp_noise_rb))

    def add_img_cb_handler(self, handler):
        "handler(i)"
        self.selected_image_cb.currentIndexChanged.connect(handler)

    def update_og_img(self, cv_img):
        """Updates the original image with a new opencv image"""
        qt_img = convert_cv_qt(cv_img, 600, 600)
        self.og_img_label.setPixmap(qt_img)

    def update_tr_img(self, cv_img):
        """Updates the transformed image with a new opencv image"""
        qt_img = convert_cv_qt(cv_img, 600, 600)
        self.tr_img_label.setPixmap(qt_img)

