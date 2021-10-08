#!/usr/bin/env python
# -*- coding: utf-8 -*-
import imutils
from PIL import Image
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QPalette, QPainter, QTransform
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
from PyQt5.QtWidgets import QLabel, QSizePolicy, QScrollArea, QMessageBox, QMainWindow, QMenu, QAction, \
    qApp, QFileDialog

import cv2
import numpy as np

""" __author__ = "Bruno Rodrigues, Igor Sabarense and Raphael Nogueira"
    __credits__ = ["PyQT5", "acbetter/QImageViewer.py"
    __date__ = "2021"
"""

DIGITS_LOOKUP = {
    (1, 1, 1, 0, 1, 1, 1): 0,
    (0, 0, 1, 0, 0, 1, 0): 1,
    (1, 0, 1, 1, 1, 1, 0): 2,
    (1, 0, 1, 1, 0, 1, 1): 3,
    (0, 1, 1, 1, 0, 1, 0): 4,
    (1, 1, 0, 1, 0, 1, 1): 5,
    (1, 1, 0, 1, 1, 1, 1): 6,
    (1, 0, 1, 0, 0, 1, 0): 7,
    (1, 1, 1, 1, 1, 1, 1): 8,
    (1, 1, 1, 1, 0, 1, 1): 9
}

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
        opcoes = QFileDialog.Options()

        nomeArquivo, _ = QFileDialog.getOpenFileName(self, 'QFileDialog.getOpenFileName()', '',
                                                     'Images (*.png *.jpeg *.jpg *.bmp *.gif)', options=opcoes)

        if nomeArquivo:
            # Cria openCV e inverte a imagem
            self.cv_imagem = cv2.imread(nomeArquivo)

            image = self.retornaQImage()
            self.contornoImagem()

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

    def contornoImagem(self):
       image = self.cv_imagem.copy()
       gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
       thresh = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
       kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))


       # find contours in the thresholded image, then initialize the
       # digit contours lists
       cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
       cnts = cnts[0]

       digitCnts = []

       # loop over the digit area candidates
       for c in cnts:
           # compute the bounding box of the contour
           (x, y, w, h) = cv2.boundingRect(c)

           # if the contour is sufficiently large, it must be a digit
           print('w,h =', w, h)
           cv2.rectangle(self.cv_imagem, (x, y), (x + w, y + h), (100, 255, 50), 1)
           if w >= 15 and (30 <= h <= 40):
               digitCnts.append(c)

       # sort the contours from left-to-right, then initialize the
       # actual digits themselves
       # digitCnts = contours.sort_contours(digitCnts,
       #	method="left-to-right")[0]
       digits = []

       # loop over each of the digits
       print('digitCnts', digitCnts)
       for c in digitCnts:
           # extract the digit ROI
           (x, y, w, h) = cv2.boundingRect(c)
           roi = thresh[y:y + h, x:x + w]

           # compute the width and height of each of the 7 segments
           # we are going to examine
           (roiH, roiW) = roi.shape
           (dW, dH) = (int(roiW * 0.25), int(roiH * 0.15))
           dHC = int(roiH * 0.05)

           # define the set of 7 segments
           segments = [
               ((0, 0), (w, dH)),  # top
               ((0, 0), (dW, h // 2)),  # top-left
               ((w - dW, 0), (w, h // 2)),  # top-right
               ((0, (h // 2) - dHC), (w, (h // 2) + dHC)),  # center
               ((0, h // 2), (dW, h)),  # bottom-left
               ((w - dW, h // 2), (w, h)),  # bottom-right
               ((0, h - dH), (w, h))  # bottom
           ]
           on = [0] * len(segments)

           # sort the contours from left-to-right, then initialize the
           # actual digits themselves
           # loop over the segments
           for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
               # extract the segment ROI, count the total number of
               # thresholded pixels in the segment, and then compute
               # the area of the segment
               segROI = roi[yA:yB, xA:xB]
               total = cv2.countNonZero(segROI)
               area = (xB - xA) * (yB - yA)

               # if the total number of non-zero pixels is greater than
               # 50% of the area, mark the segment as "on"
               if area > 0 and total / float(area) > 0.5:
                   on[i] = 1

           # lookup the digit and draw it on the image
           digit = DIGITS_LOOKUP[tuple(on)]
           digits.append(digit)
           cv2.rectangle(self.cv_imagem, (x, y), (x + w, y + h), (0, 255, 0), 1)
           cv2.putText(self.cv_imagem, str(digit), (x - 10, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

       # display the digits
       print(digits)



if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    ocr_app = QImageViewer()
    ocr_app.show()
    sys.exit(app.exec_())
