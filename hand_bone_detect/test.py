import sys
from PyQt5.QtWidgets import QLabel,QHBoxLayout,QWidget,QApplication,QMainWindow
from PyQt5.QtGui import QPixmap,QImage
from PIL import Image

class QPixmapDemo(QMainWindow):
    def __init__(self):
        super(QPixmapDemo, self).__init__()

        #设置窗口大小
        self.resize(400, 150)
        #设置窗口标题
        self.setWindowTitle("QPixmapDemo")

        im = Image.open("img/1548.png")
        im = im.convert("RGBA")
        data = im.tobytes("raw","RGBA")
        qim = QImage(data, im.size[0], im.size[1], QImage.Format_ARGB32)
        pix = QPixmap.fromImage(qim)

        label = QLabel(self)
        label.setPixmap(pix)

        #创建水平布局
        layout = QHBoxLayout()
        layout.addWidget(label)

        mainFrame = QWidget()
        mainFrame.setLayout(layout)
        self.setCentralWidget(mainFrame)

if  __name__ == '__main__':
    app = QApplication(sys.argv)
    main = QPixmapDemo()
    main.show()
    sys.exit(app.exec_())

