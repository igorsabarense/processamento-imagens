#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from matplotlib import pyplot as plt
import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QPalette, QPainter
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
from PyQt5.QtWidgets import QLabel, QSizePolicy, QScrollArea, QMessageBox, QMainWindow, QMenu, QAction, \
    qApp, QFileDialog


""" __author__ = "Bruno Rodrigues, Igor Sabarense e Raphael Nogueira"
    __date__ = "2021"
"""


class QImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.cv_imagem = None  # csv tem que ser atualizado
        self.imagem = QImage()  # imagem tem que ser atualizada junto a csv
        self.impressora = QPrinter()
        self.fatorEscala = 0.0
        self.rotacao = 0

        self.imagemTela = QLabel()  # imagem que aparece na tela
        self.imagemTela.setBackgroundRole(QPalette.Base)
        self.imagemTela.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.imagemTela.setScaledContents(True)

        self.areaRolagem = QScrollArea()
        self.areaRolagem.setAlignment(Qt.AlignCenter)
        self.areaRolagem.setBackgroundRole(QPalette.Dark)
        self.areaRolagem.setWidget(self.imagemTela)
        self.areaRolagem.setVisible(False)

        self.setCentralWidget(self.areaRolagem)

        self.criarAcoes()
        self.criarMenus()

        self.setWindowTitle("Processamento de Imagens - Reconhecimento ótico de caracteres ")
        self.resize(800, 600)

    def acaoAbrirArquivo(self):
        options = QFileDialog.Options()

        nomeArquivo, _ = QFileDialog.getOpenFileName(self, 'QFileDialog.getOpenFileName()', '',
                                                     'Images (*.png *.jpeg *.jpg *.bmp *.gif)', options=options)

        if nomeArquivo:
            self.cv_imagem = cv2.imread(nomeArquivo)

            image = self.retornaQImage()
            self.processarImagem()

            if image.isNull():
                QMessageBox.information(self, "Visualizador", "Não foi possível carregar %s." % nomeArquivo)
                return

            self.imagemTela.setPixmap(QPixmap.fromImage(image))
            self.fatorEscala = 1.0

            self.areaRolagem.setVisible(True)
            self.acaoImprimir.setEnabled(True)
            self.acaoAjustarATela.setEnabled(True)
            self.atualizarAcoes()

            if not self.acaoAjustarATela.isChecked():
                self.imagemTela.adjustSize()

    def retornaQImage(self):
        if self.cv_imagem is not None:
            # Converter formato de BGR pra RGB ( QIMAGE ) 
            self.cv_imagem = cv2.cvtColor(self.cv_imagem, cv2.COLOR_BGR2RGB)
            self.imagem = self.criarDadosQImage()
        return self.imagem

    def criarDadosQImage(self):
        altura = self.cv_imagem.shape[0]
        largura = self.cv_imagem.shape[1]
        canalCores = 3 * largura
        return QImage(self.cv_imagem.data, largura, altura, canalCores, QImage.Format_RGB888)

    def transformaImagemEmOpenCV(self):
        imagemFormatada = self.imagem.convertToFormat(4)

        largura = imagemFormatada.largura()
        altura = imagemFormatada.altura()

        ptr = imagemFormatada.bits()
        ptr.setsize(imagemFormatada.byteCount())
        return np.array(ptr).reshape(altura, largura, 4)

    def acaoImprimir(self):
        dialog = QPrintDialog(self.impressora, self)
        if dialog.exec_():
            telaImpressa = QPainter(self.impressora)
            rect = telaImpressa.viewport()
            tamanho = self.imagemTela.pixmap().size()
            tamanho.scale(rect.size(), Qt.KeepAspectRatio)
            telaImpressa.setViewport(rect.x(), rect.y(), tamanho.largura(), tamanho.altura())
            telaImpressa.setWindow(self.imagemTela.pixmap().rect())
            telaImpressa.drawPixmap(0, 0, self.imagemTela.pixmap())

    def zoomIn(self):
        self.escalarImagem(1.25)

    def zoomOut(self):
        self.escalarImagem(0.8)

    def normalSize(self):
        self.imagemTela.adjustSize()
        self.fatorEscala = 1.0

    def acaoAjustarATela(self):
        acaoAjustarATela = self.acaoAjustarATela.isChecked()
        self.areaRolagem.setWidgetResizable(acaoAjustarATela)
        if not acaoAjustarATela:
            self.normalSize()

        self.atualizarAcoes()

    def criarAcoes(self):
        self.acaoAbrirArquivo = QAction("&Abrir...", self, shortcut="Ctrl+O", triggered=self.acaoAbrirArquivo)
        self.acaoImprimir = QAction("&Imprimir...", self, shortcut="Ctrl+P", enabled=False, triggered=self.acaoImprimir)
        self.acaoSair = QAction("&Sair", self, shortcut="Ctrl+Q", triggered=self.close)
        self.zoomInAct = QAction("Zoom &In (25%)", self, shortcut="Ctrl++", enabled=False, triggered=self.zoomIn)
        self.zoomOutAct = QAction("Zoom &Out (25%)", self, shortcut="Ctrl+-", enabled=False, triggered=self.zoomOut)
        self.normalSizeAct = QAction("&Tamanho original", self, shortcut="Ctrl+S", enabled=False,
                                     triggered=self.normalSize)
        self.acaoAjustarATela = QAction("&Ajustar a tela", self, enabled=False, checkable=True, shortcut="Ctrl+F",
                                        triggered=self.acaoAjustarATela)
        self.sobrePyQT5 = QAction("Py&Qt5", self, triggered=qApp.aboutQt)

    def criarMenus(self):
        # Menu Arquivo
        self.menuArquivo = QMenu("&Arquivo", self)
        self.menuArquivo.addAction(self.acaoAbrirArquivo)
        self.menuArquivo.addAction(self.acaoImprimir)
        self.menuArquivo.addSeparator()
        self.menuArquivo.addAction(self.acaoSair)

        # Vizualizar
        self.menuVisualizar = QMenu("&Visualizar", self)
        self.menuVisualizarZoom = self.menuVisualizar.addMenu("&Zoom")
        self.menuVisualizarZoom.addAction(self.zoomInAct)
        self.menuVisualizarZoom.addAction(self.zoomOutAct)
        self.menuVisualizar.addAction(self.normalSizeAct)
        self.menuVisualizar.addSeparator()
        self.menuVisualizar.addAction(self.acaoAjustarATela)

        # Menu Processamento

        self.menuProcessamento = QMenu("&Processamento", self)

        # Help menu
        self.helpMenu = QMenu("&Sobre", self)
        self.helpMenu.addAction(self.sobrePyQT5)

        # Menu Bar
        self.menuBar().addMenu(self.menuArquivo)
        self.menuBar().addMenu(self.menuVisualizar)
        self.menuBar().addMenu(self.menuProcessamento)
        self.menuBar().addMenu(self.helpMenu)

    def atualizarAcoes(self):
        self.zoomInAct.setEnabled(not self.acaoAjustarATela.isChecked())
        self.zoomOutAct.setEnabled(not self.acaoAjustarATela.isChecked())
        self.normalSizeAct.setEnabled(not self.acaoAjustarATela.isChecked())

    def escalarImagem(self, escala):
        self.fatorEscala *= escala
        self.imagemTela.resize(self.fatorEscala * self.imagemTela.pixmap().size())

        self.ajustarBarraRolagem(self.areaRolagem.horizontalScrollBar(), escala)
        self.ajustarBarraRolagem(self.areaRolagem.verticalScrollBar(), escala)

        self.zoomInAct.setEnabled(self.fatorEscala < 3.0)
        self.zoomOutAct.setEnabled(self.fatorEscala > 0.333)

    def ajustarBarraRolagem(self, barraRolagem, escala):
        barraRolagem.setValue(int(escala * barraRolagem.value()
                                  + ((escala - 1) * barraRolagem.pageStep() / 2)))

    def sort_contours(self, cnts):
        # initialize the reverse flag and sort index
        i = 0
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                            key=lambda b: b[1][i]))

        return cnts

    def get_horizontal_projection(self, img):
        thresh = cv2.threshold(img.copy(), 130, 255,
                                cv2.THRESH_BINARY)[1]  # Change points between image binarization (130255) to 255 (background)
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

    def get_vertical_projection(self, img):
        thresh = cv2.threshold(img.copy(), 130, 255,
                                cv2.THRESH_BINARY)[1]  # Change points between image binarization (130255) to 255 (background)
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

    # a preprocess function
    def infer_prec(self,img, img_size):
        img = tf.expand_dims(img, -1)  # from 28 x 28 to 28 x 28 x 1
        img = tf.divide(img, 255)  # normalize
        img = tf.image.resize(img,  # resize acc to the input
                              [img_size, img_size])
        img = tf.reshape(img,  # reshape to add batch dimension
                         [1, img_size, img_size, 1])
        return img

    def find_white_background(self, imgArr, threshold=0.1):
        """remove images with transparent or white background"""

        background = np.array([255, 255, 255])
        percent = (imgArr == background).sum() / imgArr.size

        if percent >= threshold or percent == 0 or percent <= 0.001:
            return True
        else :
            return False

    def processarImagem(self):
       image = self.cv_imagem.copy()
       white_background = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU if self.find_white_background(image) else cv2.THRESH_BINARY + cv2.THRESH_OTSU
       gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
       blur = cv2.GaussianBlur(gray, (5, 5), 0)
       thresh = cv2.threshold(blur, 0, 255, white_background)[1]


       # find contours in the thresholded image, then initialize the
       # digit contours lists
       cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
       cnts = self.sort_contours(cnts)

       i = 0

       digits = []

       # loop over the digit area candidates
       for c in cnts:
           # compute the bounding box of the contour
           (x, y, w, h) = cv2.boundingRect(c)
           if w >= 5:
               # Taking ROI of the cotour
               roi = thresh.copy()[y:y + h, x:x + w]

               roi = np.pad(roi, ((5, 5), (5, 5)), "constant", constant_values=0)

               grayPadded = np.pad(gray[y:y + h, x:x + w], ((5, 5), (5, 5)), "constant", constant_values=0)

               v_proj  = self.get_vertical_projection(cv2.resize(grayPadded, (28,28)));
               h_proj = self.get_horizontal_projection(cv2.resize(grayPadded, (28, 28)));
               vh_proj = v_proj + h_proj

               if i is 0:
                plt.subplot(330 + 1 + i)
                plt.imshow(roi, cmap='gray')
                plt.subplot(330 + 1 + 1)
                plt.imshow(v_proj, cmap='gray')
                plt.subplot(330 + 1 + 2)
                plt.imshow(h_proj, cmap='gray')
                plt.subplot(330 + 1 + 5)
                plt.imshow(vh_proj, cmap='gray')


               tf_img = self.infer_prec(vh_proj, 28)  # call preprocess function


               prediction = TensorFlowModel.model.predict(tf_img)
               digits.append(np.argmax(prediction))

               #print(prediction)
               #append.predict

               if w < (7 * h):
                   cv2.rectangle(self.cv_imagem, (x, y), (x + w, y + h), (0, 255, 0), 1)

               i = i+1

       plt.show()
       print(digits)

class TensorFlowModel():
    def __init__(self):
        super().__init__()

    def project_dataset(dataset):
        new_data = []
        for img in dataset:

            def get_horizontal_projection(img):
                (h, w) = img.shape  # Return height and width
                a = [0 for z in range(0, w)]

                for j in range(0, h):
                    for i in range(0, w):
                        if img[j, i] == 0:
                            a[j] += 1
                            img[j, i] = 255

                for j in range(0, h):
                    for i in range(0, a[j]):
                        img[j, i] = 0

                return img

            def get_vertical_projection(img):
                (h, w) = img.shape  # Return height and width
                a = [0 for z in range(0, w)]
                # Record the peaks of each column
                for j in range(0, w):  # Traversing a column
                    for i in range(0, h):  # Traverse a row
                        if img[i, j] == 0:  # If you change the point to black
                            a[j] += 1  # Counter of this column plus one count
                            img[i, j] = 255  # Turn it white after recording

                for j in range(0, w):  # Traverse each column
                    for i in range((h - a[j]),
                                   h):  # Start from the top point of the column that should be blackened to the bottom
                        img[i, j] = 0  # Blackening

                return img
            if img.copy().shape:
                new_data.append(get_horizontal_projection(img.copy()) + get_vertical_projection(img.copy()))
        return new_data


    mnist = tf.keras.datasets.mnist
    (x_treino , y_treino) , (x_teste, y_teste) = mnist.load_data()

    x_treino = tf.keras.utils.normalize(x_treino, axis= 1)
    x_teste = tf.keras.utils.normalize(x_teste, axis= 1)

    custom_preprocessing_x_train = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=project_dataset(x_treino)).preprocessing_function
    custom_preprocessing_x_train = tf.keras.utils.normalize(custom_preprocessing_x_train, axis= 1)
    plt.imshow(custom_preprocessing_x_train[0], cmap="gray")
    plt.show()

    custom_preprocessing_y_train = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=project_dataset(y_treino)).preprocessing_function
    custom_preprocessing_y_train = tf.keras.utils.normalize(custom_preprocessing_x_train, axis=1)
    plt.imshow(custom_preprocessing_y_train[0], cmap="gray")
    plt.show()

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=128, activation= tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))
    model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(custom_preprocessing_x_train, custom_preprocessing_y_train,  epochs = 3)
    perda, acuracia  = model.evaluate(x_teste, y_teste)
    print('accuracy ' , acuracia)
    print('loss', perda)
    model.summary()
    model.save('ocr.model')




if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    ocr_app = QImageViewer()
    ocr_app.show()
    sys.exit(app.exec_())
