#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" __author__ = "Bruno Rodrigues, Igor Sabarense e Raphael Nogueira"
    __date__ = "2021"
"""

import pickle
import cv2
import numpy as np
import tensorflow as tf
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QPalette, QPainter, QIcon
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
from PyQt5.QtWidgets import QLabel, QSizePolicy, QScrollArea, QMessageBox, QMainWindow, QMenu, QAction, \
    qApp, QFileDialog

from imutils import contours
from keras.utils.np_utils import normalize
from matplotlib import pyplot as plt

import ann
import svm

from processing_utils import deskew, getVerticalProjectionProfile, getHorizontalProjectionProfile, \
    interpolate_projection, find_white_background, resize_image


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

    def update_scrolling_area(self, scroll_area, scale):
        scroll_area.setValue(int(scale * scroll_area.value()
                                 + ((scale - 1) * scroll_area.pageStep() / 2)))

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

        self.update_scrolling_area(self.scroll_area.horizontalScrollBar(), scale)
        self.update_scrolling_area(self.scroll_area.verticalScrollBar(), scale)

        self.zoom_in.setEnabled(self.scale < 3.0)
        self.zoom_out.setEnabled(self.scale > 0.333)


    """ 
         Above this line, the methods are interface default methods.
         Below this line , the methods are used to create the OCR.
    """

    def svm(self):
        # carrega o modelo svm
        model = None
        try:
            model = pickle.load(open("svm_model.sav", 'rb'))
        except:
            svm.run_support_vector_machine_model()
            model = pickle.load(open("svm_model.sav", 'rb'))
        finally:
            self.draw_prediction(model, "SVM ( Support Vector Machine )", False)

    def artificial_neural_network(self):
        # carrega o modelo rede neural
        model = None
        try:
            model = tf.keras.models.load_model("neural_network")
        except:
            ann.run_neural_network_model()
            model = tf.keras.models.load_model("neural_network")
        finally:
            self.draw_prediction(model, "Rede Neural Artificial", True)



    # desenha na tela o resultado da IA1
    def draw_prediction(self, model, title, is_ann):

        # projecoes
        digits = np.array(self.projections)
        # regiao de interesse
        rois = self.roi_digits

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
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # transforma em escala de cinza
        blur = cv2.GaussianBlur(gray, (5, 5), 0)  # embaça para tirar ruído
        thresh = cv2.threshold(blur, 0, 255, white_background)[1]  # limiarização da imagem

        # find contours in the thresholded image, then initialize the
        # digit contours lists
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)  # acha os contornos dos digitos na imagem
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cnts, _ = contours.sort_contours(cnts, method="left-to-right")  # ordena os contornos da esquerda para direita

        i = 0

        # loop over the digit area candidates
        for c in cnts:
            # compute the bounding box of the contour
            (x, y, w, h) = cv2.boundingRect(c)

            if w >= 5 and h >= 10:
                # Taking ROI of the cotour
                # MNIST 20x20 centered in a bounding box 28x28
                roi = thresh.copy()[y:y + h, x:x + w]  # pega a regiao de interesse da imagem
                roi = deskew(roi)  # alinha a imagem para ficar reta
                roi = resize_image(roi)  # ajusta o tamanho da imagem para 18,18 para depois ficar mais proxima ao MNIST
                roi = np.pad(roi, ((5, 5), (5, 5)), "constant",
                             constant_values=0)  # centraliza a imagem assim transformando em 28,28

                cv2.rectangle(self.cv_image, (x, y), (x + w, y + h), (0, 255, 0),
                              2)  # cria um retangulo verde demonstrando os digitos

                v_proj = getVerticalProjectionProfile(roi)  # cria projecao vertical
                h_proj = getHorizontalProjectionProfile(roi)  # cria projecao horizontal
                vh_proj = interpolate_projection(v_proj) + interpolate_projection(h_proj)  # concatena as duas projecoes
                vh_proj = normalize(vh_proj, axis=0)[0]

                self.roi_digits.append(roi)  # adiciona o digito a um vetor de regioes de interesse
                self.projections.append(vh_proj)  # adiciona a projecao concatenada a um vetor
                i = i + 1


if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    ocr_app = App()
    ocr_app.show()
    sys.exit(app.exec_())
