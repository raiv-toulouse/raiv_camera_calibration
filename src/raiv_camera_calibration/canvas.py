from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *


class Canvas(QWidget):

    def __init__(self,parent):
        super().__init__(parent)
        self.parent = parent
        self.image = None

    def set_image(self, q_image):
        self.image = q_image
        self.setMinimumSize(self.image.width(), self.image.height())
        self.update()

    def paintEvent(self, event):
        if self.image:
            qp = QPainter(self)
            rect = event.rect()
            qp.drawImage(rect, self.image, rect)
            # if self.parent.cx != None: # Draw point (cx,cy) when it has been calculated
            #     pen = QPen(Qt.red, 5)
            #     qp.setPen(pen)
            #     x_center = int(self.parent.cx)
            #     y_center = int(self.parent.cy)
            #     qp.drawPoint(x_center, y_center)
            qp.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            pos = event.pos()
            pixel_coord = [pos.x(), pos.y()]
            self.parent.pixel_coord = pixel_coord



