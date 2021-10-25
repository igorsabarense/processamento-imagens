#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import cv2
import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QPalette, QPainter, QIcon
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
from PyQt5.QtWidgets import QLabel, QSizePolicy, QScrollArea, QMessageBox, QMainWindow, QMenu, QAction, \
    qApp, QFileDialog
from matplotlib import pyplot as plt
from scipy.ndimage import interpolation
from tensorflow.keras.utils import to_categorical
from imutils import contours

import pydot
import seaborn as sns

# Evaluation library
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

# Digit MNIST dataset
(X_train_digit, y_train_digit), (X_test_digit, y_test_digit) = mnist.load_data()
# Encoding Digit MNIST Labels
y_train_digit = to_categorical(y_train_digit)
y_test_digit = to_categorical(y_test_digit)

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


def find_white_background(imgArr, threshold=0.1815):
    """remove images with transparent or white background"""
    background = np.array([255, 255, 255])
    percent = (imgArr == background).sum() / imgArr.size
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


def resize_image(img, size=(18, 18)):
    h, w = img.shape[:2]
    c = img.shape[2] if len(img.shape) > 2 else 1

    if h == w:
        return cv2.resize(img, size, cv2.INTER_AREA)

    dif = h if h > w else w

    interpolation = cv2.INTER_AREA if dif > (size[0] + size[1]) // 2 else cv2.INTER_CUBIC

    x_pos = (dif - w) // 2
    y_pos = (dif - h) // 2

    if len(img.shape) == 2:
        mask = np.zeros((dif, dif), dtype=img.dtype)
        mask[y_pos:y_pos + h, x_pos:x_pos + w] = img[:h, :w]
    else:
        mask = np.zeros((dif, dif, c), dtype=img.dtype)
        mask[y_pos:y_pos + h, x_pos:x_pos + w, :] = img[:h, :w, :]

    return cv2.resize(mask, size, interpolation)


def moments(image):
    c0, c1 = np.mgrid[:image.shape[0], :image.shape[1]]  # A trick in numPy to create a mesh grid
    totalImage = np.sum(image)  # sum of pixels
    m0 = np.sum(c0 * image) / totalImage  # mu_x
    m1 = np.sum(c1 * image) / totalImage  # mu_y
    m00 = np.sum((c0 - m0) ** 2 * image) / totalImage  # var(x)
    m11 = np.sum((c1 - m1) ** 2 * image) / totalImage  # var(y)
    m01 = np.sum((c0 - m0) * (c1 - m1) * image) / totalImage  # covariance(x,y)
    mu_vector = np.array([m0, m1])  # Notice that these are \mu_x, \mu_y respectively
    covariance_matrix = np.array([[m00, m01], [m01, m11]])  # Do you see a similarity between the covariance matrix
    return mu_vector, covariance_matrix


def deskew(image):
    c, v = moments(image)
    alpha = v[0, 1] / v[0, 0]
    affine = np.array([[1, 0], [alpha, 1]])
    ocenter = np.array(image.shape) / 2.0
    offset = c - np.dot(affine, ocenter)
    return interpolation.affine_transform(image, affine, offset=offset)


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
        self.setWindowIcon(QIcon('logo_pucminas.png'))

        self.setWindowTitle("Processamento de Imagens - Reconhecimento ótico de caracteres ")
        self.resize(1024, 768)

        self.cv_image = None
        self.roi_digits = []
        self.projections = []

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
        self.ann = QAction("&Rede Neural Artificial", self, enabled=False, triggered=self.artificial_neural_network)
        self.svm = QAction("&SVM", self, enabled=False, triggered=self.svm)
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
        self.menu_processing.addAction(self.ann)
        self.menu_processing.addAction(self.svm)
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
        self.ann.setEnabled(not self.fit_canvas.isChecked())
        self.svm.setEnabled(not self.fit_canvas.isChecked())

    def scale_canvas_image(self, scale):
        self.scale *= scale
        self.canvas_image.resize(self.scale * self.canvas_image.pixmap().size())

        update_scrolling_area(self.scroll_area.horizontalScrollBar(), scale)
        update_scrolling_area(self.scroll_area.verticalScrollBar(), scale)

        self.zoom_in.setEnabled(self.scale < 3.0)
        self.zoom_out.setEnabled(self.scale > 0.333)

    def svm(self):
        self.draw_prediction("svm_model.sav", "SVM ( Support Vector Machine )", False)

    def artificial_neural_network(self):
        self.draw_prediction("neural_network", "Rede Neural Artificial", True)


    #desenha na tela o resultado da IA1
    def draw_prediction(self, model_name, title, is_ann):
        #carrega o modelo svm ou rede neural
        model = tf.keras.models.load_model(model_name) if is_ann else pickle.load(open(model_name, 'rb'))
        #projecoes
        digits = np.array(self.projections)
        #regiao de interesse
        rois = self.roi_digits

        # utiliza o modelo para descobrir quais são os digitos de acordo com o vetor de projecoes
        if not is_ann:
            digits = digits.reshape(digits.shape[0], 28 * 28)

        prediction = model.predict(digits)

        if is_ann:
            prediction = np.argmax(prediction, axis=1)

        # plota uma imagem com os resultados de sua IA
        fig = plt.figure(figsize=(8, 6))
        for i in range(len(rois)):
            plt.subplot(8, 6, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(rois[i].reshape(28, 28), cmap='gray')
            plt.xlabel(prediction[i])
        plt.subplots_adjust(hspace=0.5)
        plt.suptitle(title)
        plt.show()

        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        self.cv_image = image_from_plot
        image = self.return_canvas_image()

        self.canvas_image.setPixmap(QPixmap.fromImage(image))
        self.scale = 1.0

        self.scroll_area.setVisible(True)
        self.act_print.setEnabled(True)
        self.fit_canvas.setEnabled(False)
        self.update_canvas()

        if not self.fit_canvas.isChecked():
            self.canvas_image.adjustSize()

    def process_image(self):
        self.roi_digits = []
        self.projections = []
        image = self.cv_image.copy()  # cria copia da imagem na tela
        white_background = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU if find_white_background(
            image) else cv2.THRESH_BINARY + cv2.THRESH_OTSU  # de acordo com o fundo da imagem, cria um metodo de segmentação diferente
                                                             # fundo preto ou branco
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)       # transforma em escala de cinza
        blur = cv2.GaussianBlur(gray, (5, 5), 0)             # embaça para tirar ruído
        thresh = cv2.threshold(blur, 0, 255, white_background)[1]   #limiarização da imagem

        # find contours in the thresholded image, then initialize the
        # digit contours lists
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #acha os contornos dos digitos na imagem
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cnts, _ = contours.sort_contours(cnts, method="left-to-right") #ordena os contornos da esquerda para direita

        i = 0

        # loop over the digit area candidates
        for c in cnts:
            # compute the bounding box of the contour
            (x, y, w, h) = cv2.boundingRect(c)

            if w >= 5 and h >= 10:
                # Taking ROI of the cotour
                # MNIST 20x20 centered in a bounding box 28x28
                roi = thresh.copy()[y:y + h, x:x + w]      #pega a regiao de interesse da imagem
                roi = deskew(roi)                          # alinha a imagem para ficar reta
                roi = resize_image(roi)                    # ajusta o tamanho da imagem para 18,18 para depois ficar mais proxima ao MNIST
                roi = np.pad(roi, ((5, 5), (5, 5)), "constant", constant_values=0)  # centraliza a imagem assim transformando em 28,28

                cv2.rectangle(self.cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)    # cria um retangulo verde demonstrando os digitos

                v_proj = get_vertical_projection(roi)           #cria projecao vertical
                h_proj = get_horizontal_projection(roi)         #cria projecao horizontal
                vh_proj = v_proj + h_proj                       #concatena as duas projecoes

                reshaped_projection = np.array(vh_proj).reshape(1, 28, 28)  #reshape para se adequar ao modelo

                self.roi_digits.append(roi)                        # adiciona o digito a um vetor de regioes de interesse
                self.projections.append(reshaped_projection)       # adiciona a projecao concatenada a um vetor
                i = i + 1


if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    ocr_app = App()
    ocr_app.show()
    sys.exit(app.exec_())
