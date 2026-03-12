import sys
import typing

from PyQt5.QtWidgets import *
from PyQt5 import QtCore, uic
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPixmap,QImage
from detect_utils import load_model,detect_img
from PIL import Image

class MyWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        #加载模型
        self.yolov5_model,self.cls_models = load_model()
        self.init_ui()
    
    def init_ui(self):
        self.ui = uic.loadUi("bone_detect_ui.ui")

        self.upload_btn = self.ui.upload_btn
        self.detect_btn = self.ui.detect_btn

        self.man_btn = self.ui.man_btn
        self.man_btn.setChecked(True)
        self.woman_btn = self.ui.woman_btn

        self.img_label = self.ui.img_label
        self.result_text = self.ui.result_text

        self.upload_btn.clicked.connect(self.upload_img)
        self.detect_btn.clicked.connect(self.detect_img)

    def upload_img(self):
        self.img_path,_ =QFileDialog.getOpenFileName(self,"open file","c:\\",'Image files (*.jpg *.png *.jpeg)')
        #PIL 转 QPixmap
        im = Image.open(self.img_path)
        im = im.resize((681,681))
        im = im.convert("RGBA")
        data = im.tobytes("raw","RGBA")
        qim = QImage(data, im.size[0], im.size[1], QImage.Format_ARGB32)
        pixmap = QPixmap.fromImage(qim)
        self.img_label.setPixmap(pixmap)
    
    def detect_img(self):
        sex = "boy" if self.man_btn.isChecked() else "girl"
        export = detect_img(self.yolov5_model,self.cls_models,self.img_path,sex)
        self.result_text.setText(export)

if __name__== "__main__":
  app = QApplication(sys.argv)
  w = MyWindow()
  w.ui.show()
  app.exec()
