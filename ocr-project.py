#!/usr/bin/env python
# -*- coding: utf-8 -*-


import cv2
import numpy as np
import tensorflow as tf
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QPalette, QPainter
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
from PyQt5.QtWidgets import QLabel, QSizePolicy, QScrollArea, QMessageBox, QMainWindow, QMenu, QAction, \
    qApp, QFileDialog
from matplotlib import pyplot as plt

""" __author__ = "Bruno Rodrigues, Igor Sabarense e Raphael Nogueira"
    __date__ = "2021"
"""


def get_vertical_projection(img):
    thresh = cv2.threshold(img.copy(), 130, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    (h, w) = thresh.shape  # Return height and width
    a = [0 for z in range(0, w)]
    # Record the peaks of each column
    for j in range(0, w):  # Traversing a column
        for i in range(0, h):  # Traverse a row
            if thresh[i, j] == 0:  # If you change the point to black
                a[j] += 1  # Counter of this column plus one count
                thresh[i, j] = 255  # Turn it white after recording

    for j in range(0, w):  # Traverse each column
        for i in range((h - a[j]),
                       h):  # Start from the top point of the column that should be blackened to the bottom
            thresh[i, j] = 0  # Blackening

    return thresh


def get_horizontal_projection(img):
    thresh = cv2.threshold(img.copy(), 130, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    (h, w) = thresh.shape  # Return height and width
    a = [0 for z in range(0, w)]

    for j in range(0, h):
        for i in range(0, w):
            if thresh[j, i] == 0:
                a[j] += 1
                thresh[j, i] = 255

    for j in range(0, h):
        for i in range(0, a[j]):
            thresh[j, i] = 0

    return thresh


def infer_prec(img, img_size):
    img = tf.expand_dims(img, -1)  # from 28 x 28 to 28 x 28 x 1
    img = tf.divide(img, 255)  # normalize
    img = tf.image.resize(img,  # resize acc to the input
                          [img_size, img_size])
    img = tf.reshape(img,  # reshape to add batch dimension
                     [1, img_size, img_size, 1])
    return img


def find_white_background(imgArr, threshold=0.1815):
    """remove images with transparent or white background"""
    background = np.array([255, 255, 255])
    percent = (imgArr == background).sum() / imgArr.size
    print(percent * 100, 'branco')
    if percent >= threshold or percent == 0 or percent <= 0.001:
        return True
    else:
        return False


def update_scrolling_area(scroll_area, scale):
    scroll_area.setValue(int(scale * scroll_area.value()
                              + ((scale - 1) * scroll_area.pageStep() / 2)))


def sort_contours(cnts):
    # initialize the reverse flag and sort index
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i]))

    return cnts


class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.image = QImage()
        self.printer = QPrinter()
        self.scale = 0.0
        self.rot = 0

        self.canvas_image = QLabel()
        self.canvas_image.setBackgroundRole(QPalette.Base)
        self.canvas_image.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.canvas_image.setScaledContents(True)

        self.scroll_area = QScrollArea()
        self.scroll_area.setAlignment(Qt.AlignCenter)
        self.scroll_area.setBackgroundRole(QPalette.Dark)
        self.scroll_area.setWidget(self.canvas_image)
        self.scroll_area.setVisible(False)

        self.setCentralWidget(self.scroll_area)

        self.canvas_actions()
        self.canvas_menu()

        self.setWindowTitle("Processamento de Imagens - Reconhecimento ótico de caracteres ")
        self.resize(800, 600)

        self.cv_image = None

    def open_file(self):
        options = QFileDialog.Options()

        filename, _ = QFileDialog.getOpenFileName(self, 'QFileDialog.getOpenFileName()', '',
                                                   'Images (*.png *.jpeg *.jpg *.bmp *.gif)', options=options)

        if filename:
            self.cv_image = cv2.imread(filename)

            image = self.return_canvas_image()
            self.process_image()

            if image.isNull():
                QMessageBox.information(self, "Visualizador", "Não foi possível carregar %s." % filename)
                return

            self.canvas_image.setPixmap(QPixmap.fromImage(image))
            self.scale = 1.0

            self.scroll_area.setVisible(True)
            self.act_print.setEnabled(True)
            self.fit_canvas.setEnabled(True)
            self.update_canvas()

            if not self.fit_canvas.isChecked():
                self.canvas_image.adjustSize()

    def return_canvas_image(self):
        if self.cv_image is not None:
            # Converter formato de BGR pra RGB ( QIMAGE )
            self.cv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
            self.image = self.return_canvas_image_data()
        return self.image

    def return_canvas_image_data(self):
        height = self.cv_image.shape[0]
        width = self.cv_image.shape[1]
        color_channel = 3 * width
        return QImage(self.cv_image.data, width, height, color_channel, QImage.Format_RGB888)

    def canvas_image_to_cv(self):
        form_image = self.image.convertToFormat(4)

        width = form_image.width()
        height = form_image.height()

        ptr = form_image.bits()
        ptr.setsize(form_image.byteCount())
        return np.array(ptr).reshape(height, width, 4)

    def act_print(self):
        dialog = QPrintDialog(self.impressora, self)
        if dialog.exec_():
            printed_canvas = QPainter(self.impressora)
            rect = printed_canvas.viewport()
            canvas_size = self.canvas_image.pixmap().size()
            canvas_size.scale(rect.size(), Qt.KeepAspectRatio)
            printed_canvas.setViewport(rect.x(), rect.y(), canvas_size.width(), canvas_size.height())
            printed_canvas.setWindow(self.canvas_image.pixmap().rect())
            printed_canvas.drawPixmap(0, 0, self.canvas_image.pixmap())

    def zoomIn(self):
        self.scale_canvas_image(1.25)

    def zoomOut(self):
        self.scale_canvas_image(0.8)

    def normal_size(self):
        self.canvas_image.adjustSize()
        self.scale = 1.0

    def fit_canvas(self):
        fit_canvas = self.fit_canvas.isChecked()
        self.scroll_area.setWidgetResizable(fit_canvas)
        if not fit_canvas:
            self.normal_size()

        self.update_canvas()

    def canvas_actions(self):
        self.open_file = QAction("&Abrir...", self, shortcut="Ctrl+O", triggered=self.open_file)
        self.act_print = QAction("&Imprimir...", self, shortcut="Ctrl+P", enabled=False, triggered=self.act_print)
        self.exit = QAction("&Sair", self, shortcut="Ctrl+Q", triggered=self.close)
        self.zoom_in = QAction("Zoom &In (25%)", self, shortcut="Ctrl++", enabled=False, triggered=self.zoomIn)
        self.zoom_out = QAction("Zoom &Out (25%)", self, shortcut="Ctrl+-", enabled=False, triggered=self.zoomOut)
        self.normal_size = QAction("&Tamanho original", self, shortcut="Ctrl+S", enabled=False,
                                     triggered=self.normal_size)
        self.fit_canvas = QAction("&Ajustar a tela", self, enabled=False, checkable=True, shortcut="Ctrl+F",
                                  triggered=self.fit_canvas)
        self.sobrePyQT5 = QAction("Py&Qt5", self, triggered=qApp.aboutQt)

    def canvas_menu(self):
        # Menu Arquivo
        self.menu_file = QMenu("&Arquivo", self)
        self.menu_file.addAction(self.open_file)
        self.menu_file.addAction(self.act_print)
        self.menu_file.addSeparator()
        self.menu_file.addAction(self.exit)

        # Vizualizar
        self.menu_view = QMenu("&Visualizar", self)
        self.menu_view_zoom = self.menu_view.addMenu("&Zoom")
        self.menu_view_zoom.addAction(self.zoom_in)
        self.menu_view_zoom.addAction(self.zoom_out)
        self.menu_view.addAction(self.normal_size)
        self.menu_view.addSeparator()
        self.menu_view.addAction(self.fit_canvas)

        # Menu Processamento

        self.menu_processing = QMenu("&Processamento", self)

        # Help menu
        self.help_menu = QMenu("&Sobre", self)
        self.help_menu.addAction(self.sobrePyQT5)

        # Menu Bar
        self.menuBar().addMenu(self.menu_file)
        self.menuBar().addMenu(self.menu_view)
        self.menuBar().addMenu(self.menu_processing)
        self.menuBar().addMenu(self.help_menu)

    def update_canvas(self):
        self.zoom_in.setEnabled(not self.fit_canvas.isChecked())
        self.zoom_out.setEnabled(not self.fit_canvas.isChecked())
        self.normal_size.setEnabled(not self.fit_canvas.isChecked())

    def scale_canvas_image(self, scale):
        self.scale *= scale
        self.canvas_image.resize(self.scale * self.canvas_image.pixmap().size())

        update_scrolling_area(self.scroll_area.horizontalScrollBar(), scale)
        update_scrolling_area(self.scroll_area.verticalScrollBar(), scale)

        self.zoom_in.setEnabled(self.scale < 3.0)
        self.zoom_out.setEnabled(self.scale > 0.333)

    def process_image(self):
        image = self.cv_image.copy()
        white_background = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU if find_white_background(
            image) else cv2.THRESH_BINARY + cv2.THRESH_OTSU
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blur, 0, 255, white_background)[1]

        # find contours in the thresholded image, then initialize the
        # digit contours lists
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        cnts = sort_contours(cnts)

        i = 0

        digits = []

        model = tf.keras.models.load_model('neural_network')

        # loop over the digit area candidates
        for c in cnts:
            # compute the bounding box of the contour
            (x, y, w, h) = cv2.boundingRect(c)

            if w >= 6 and h >= 10:
                # Taking ROI of the cotour
                roi = thresh.copy()[y:y + h, x:x + w]
                roi = np.pad(roi, ((5, 5), (5, 5)), "constant", constant_values=0)
                roi = cv2.resize(roi, (28, 28))

                cv2.rectangle(self.cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                if i == 0:
                    plt.imshow(roi, cmap="gray")

                v_proj = get_vertical_projection(roi)
                h_proj = get_horizontal_projection(roi)
                vh_proj = v_proj + h_proj

                prediction = model.predict(np.array(vh_proj).reshape(1, 28, 28))
                digits.append(np.argmax(prediction[0]))

                i = i + 1

        plt.show()
        print(digits)


if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    ocr_app = App()
    ocr_app.show()
    sys.exit(app.exec_())
